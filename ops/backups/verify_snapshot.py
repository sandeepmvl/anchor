import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anchor_ai.storage import initialize_sandbox, recovery_inventory
from anchor_ai.index_maintenance import verify_recovery_index, retrieval_snapshot
from ops.deploy.release_bundle import identity, verify_recovered_releases
import yaml

HOME = ROOT / 'tmp/anchor-recovery'
LIMITATIONS = ['Rails requests not verified', 'payment-provider reconciliation not verified',
               'off-host durability not verified', 'production writer coordination not verified']


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def policy():
    p = yaml.safe_load((ROOT / 'ops/backups/policy.yml').read_text())
    if p['version'] != 1 or min(p['frequency_seconds'], p['rpo_target_seconds'],
                               p['rto_target_seconds'], *p['retention'].values()) <= 0:
        raise ValueError('invalid_backup_policy')
    return p


# Ruby's OpenSSL is available offline. Tag authentication completes before stdout.
CRYPTO = '''require 'openssl'
key = [ENV.fetch('ANCHOR_BACKUP_KEY')].pack('H*')
raise 'invalid_backup_key' unless ENV.fetch('ANCHOR_BACKUP_KEY').match?(/\\A[0-9a-f]{64}\\z/)
c = OpenSSL::Cipher.new('aes-256-gcm')
raw = STDIN.read.b
if ARGV.fetch(0) == 'encrypt'
  c.encrypt
  c.key = key
  nonce = OpenSSL::Random.random_bytes(12)
  c.iv = nonce
  c.auth_data = 'anchor-snapshot-v1'
  encrypted = c.update(raw) + c.final
  STDOUT.binmode.write(nonce + c.auth_tag + encrypted)
else
  raise 'short_envelope' if raw.bytesize < 29
  c.decrypt
  c.key = key
  c.iv = raw.byteslice(0,12)
  c.auth_tag = raw.byteslice(12,16)
  c.auth_data = 'anchor-snapshot-v1'
  plain = c.update(raw.byteslice(28..)) + c.final
  STDOUT.binmode.write(plain)
end
'''


def crypt(raw, mode):
    result = subprocess.run(['ruby', '-e', CRYPTO, mode], input=raw,
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError('encryption_or_authentication_failed')
    return result.stdout


def connection(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    return db


def fixture(path):
    """Runtime projection, not migrations or a replacement for Rails models."""
    initialize_sandbox(path)
    with connection(path) as db:
        db.executescript('''
        ALTER TABLE document_chunks ADD COLUMN checksum TEXT;
        CREATE TABLE recovery_conversations(tenant_id INTEGER, id INTEGER, ticket TEXT,
          PRIMARY KEY(tenant_id,id));
        CREATE TABLE recovery_actions(tenant_id INTEGER, conversation_id INTEGER,
          action_key TEXT, result TEXT, PRIMARY KEY(tenant_id,conversation_id,action_key));
        CREATE TABLE recovery_effects(tenant_id INTEGER, conversation_id INTEGER, amount INTEGER);
        CREATE TABLE recovery_memory(tenant_id INTEGER, conversation_id INTEGER, value TEXT);
        CREATE TABLE recovery_releases(tenant_id INTEGER PRIMARY KEY, snapshot TEXT);
        ''')
        for tenant in (1, 2):
            text = f'Tenant {tenant} original policy'
            db.execute('''INSERT INTO documents
              (id,tenant_id,external_id,source_name,content_hash,text,status)
              VALUES(?,?,?,?,?,?,?)''', (tenant, tenant, 'policy', 'fixture', sha(text.encode()), text, 'ready'))
            db.execute('''INSERT INTO retrieval_generations
              (id,tenant_id,model,revision,dimensions,status,active,vector_column)
              VALUES(?,?, 'offline', 'v1', 2, 'ready', 1, 'embedding')''', (tenant, tenant))
            db.execute('''INSERT INTO document_chunks
              (tenant_id,document_id,retrieval_generation_id,chunk_index,is_leaf,text,
               embedding,embedding_model,embedding_revision,embedding_dimensions,
               source_hash,chunker_version,checksum)
              VALUES(?,?,?,0,1,?,'[1,0]','offline','v1',2,?,'sentence-v1',?)''',
              (tenant, tenant, tenant, text, sha(text.encode()), sha(text.encode())))
            db.execute('INSERT INTO recovery_conversations VALUES(?,1,?)', (tenant, 'refund-ticket'))
            db.execute('INSERT INTO recovery_actions VALUES(?,1,?,?)', (tenant, 'refund-1', 'completed'))
            db.execute('INSERT INTO recovery_effects VALUES(?,1,100)', (tenant,))
            db.execute('INSERT INTO recovery_memory VALUES(?,1,?)', (tenant, 'refund completed'))
            bundle = dict(contract='anchor-support-v1', index=str(tenant), model='offline',
                          offline_only=True, prompt='prompt-v1', reply=text, ticket='refund-ticket')
            snap = dict(tenant_id=tenant, generation=1, bundle_id=identity(bundle), bundle=bundle)
            db.execute('INSERT INTO recovery_releases VALUES(?,?)', (tenant, json.dumps(snap)))


def artifacts(db):
    generations = [dict(r) for r in db.execute('''SELECT tenant_id,id,model,revision,
        dimensions,vector_column FROM retrieval_generations WHERE active=1 ORDER BY tenant_id''')]
    releases = [json.loads(r[0]) for r in db.execute('SELECT snapshot FROM recovery_releases ORDER BY tenant_id')]
    return {'index.json': {'format': 1, 'generations': generations},
            'releases.json': releases,
            'prompts.json': {'prompt-v1': 'Answer only from the tenant policy.'},
            'golden.json': {'version': 'golden-v1', 'cases': ['refund-ticket']},
            'config.json': {'contract': 'anchor-support-v1', 'golden': 'golden-v1'}}


def package(path):
    with connection(path) as db:
        files = artifacts(db)
        inventory = recovery_inventory(db)
    raw = Path(path).read_bytes()
    manifest = dict(format=1, created_at=time.time(), database_sha256=sha(raw), tables=inventory,
                    artifact_sha256={k: sha(canonical(v)) for k, v in files.items()})
    return dict(database=base64.b64encode(raw).decode(), artifacts=files, manifest=manifest)


def verify(payload):
    """Return verified bytes only; never activate or overwrite a serving database."""
    manifest = payload['manifest']
    files = payload['artifacts']
    required = {'index.json', 'releases.json', 'prompts.json', 'golden.json', 'config.json'}
    if manifest['format'] != 1 or set(files) != required or set(manifest['artifact_sha256']) != required:
        raise ValueError('missing_or_unknown_artifact')
    for name in required:
        if sha(canonical(files[name])) != manifest['artifact_sha256'][name]:
            raise ValueError('tampered_artifact')
    raw = base64.b64decode(payload['database'], validate=True)
    if sha(raw) != manifest['database_sha256']:
        raise ValueError('database_hash_mismatch')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'candidate.sqlite3'
        path.write_bytes(raw)
        with connection(path) as db:
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or db.execute('PRAGMA foreign_key_check').fetchall():
                raise ValueError('database_integrity_failed')
            if recovery_inventory(db) != manifest['tables']:
                raise ValueError('record_checksum_mismatch')
            for row in db.execute('SELECT text,checksum FROM document_chunks'):
                if sha(row['text'].encode()) != row['checksum']:
                    raise ValueError('chunk_checksum_mismatch')
            generations = verify_recovery_index(db, files['index.json'])
            verify_recovered_releases(files['releases.json'], generations, offline=True)
            if artifacts(db) != files:
                raise ValueError('registry_artifact_mismatch')
            if files['config.json']['golden'] != files['golden.json']['version']:
                raise ValueError('golden_version_mismatch')
            for release in files['releases.json']:
                if release['bundle']['prompt'] not in files['prompts.json']:
                    raise ValueError('missing_prompt_version')
    return raw


def retain(directory, now=None):
    now = time.time() if now is None else now
    p = policy()['retention']
    keep, hours, days = set(), set(), set()
    points = sorted(Path(directory).glob('snapshot-*.enc'), key=lambda f: f.stat().st_mtime, reverse=True)
    for path in points:
        stamp = path.stat().st_mtime
        hour, day = int(stamp // 3600), int(stamp // 86400)
        if 0 <= now - stamp < p['hourly'] * 3600 and hour not in hours:
            keep.add(path)
            hours.add(hour)
        if 0 <= now - stamp < p['daily'] * 86400 and day not in days:
            keep.add(path)
            days.add(day)
    for path in points:
        if path not in keep:
            path.unlink()


def snapshot(offline=False):
    if not offline or os.environ.get('RAILS_ENV') == 'production':
        raise ValueError('production_coordinated_adapter_required')
    HOME.mkdir(parents=True, exist_ok=True)
    os.chmod(HOME, 0o700)
    with tempfile.TemporaryDirectory(dir=HOME) as directory:
        source = Path(directory) / 'source.sqlite3'
        fixture(source)
        # Only this process owns these fixture writers and immutable artifacts.
        payload = package(source)
        verify(payload)
    encrypted = crypt(canonical(payload), 'encrypt')
    path = HOME / f'snapshot-{time.time_ns()}.enc'
    path.write_bytes(encrypted)
    os.chmod(path, 0o600)
    (HOME / 'latest.json').write_text(json.dumps({'snapshot': path.name}))
    retain(HOME)
    print(json.dumps({'encrypted_snapshot': str(path), 'integrity_manifest': payload['manifest']}))
    return path


def guard_target(target):
    if Path(target).resolve() != (HOME / 'disposable.sqlite3').resolve() or Path(target).is_symlink():
        raise ValueError('disposable_target_required')
    if os.environ.get('RAILS_ENV') == 'production':
        raise ValueError('production_target_refused')


def drill(target=None):
    target = HOME / 'disposable.sqlite3' if target is None else Path(target)
    guard_target(target)
    started = time.monotonic()
    name = json.loads((HOME / 'latest.json').read_text())['snapshot']
    if Path(name).name != name:
        raise ValueError('invalid_snapshot_name')
    payload = json.loads(crypt((HOME / name).read_bytes(), 'decrypt'))
    raw = verify(payload)
    age = time.time() - payload['manifest']['created_at']
    p = policy()
    if age < 0 or age > p['rpo_target_seconds']:
        raise ValueError('snapshot_outside_loss_budget')
    # This target belongs solely to this drill. Never touch the source or Rails DB.
    target.unlink(missing_ok=True)
    target.write_bytes(raw)
    with connection(target) as db:
        db.execute('DELETE FROM document_chunks')
    with connection(target) as db:
        assert db.execute('SELECT count(*) FROM document_chunks').fetchone()[0] == 0
    target.write_bytes(raw)
    with connection(target) as db:
        assert recovery_inventory(db) == payload['manifest']['tables']
        for tenant in (1, 2):
            _, rows = retrieval_snapshot(db, tenant)
            assert rows and all(r['tenant_id'] == tenant for r in rows)
            assert rows[0]['text'] == f'Tenant {tenant} original policy'
        # A completed action returns its durable result, not a second effect.
        action = db.execute('SELECT result FROM recovery_actions WHERE tenant_id=1 AND conversation_id=1 AND action_key=?', ('refund-1',)).fetchone()
        assert action[0] == 'completed'
        effects = db.execute('SELECT count(*) FROM recovery_effects WHERE tenant_id=1').fetchone()[0]
        assert effects == 1
    elapsed = time.monotonic() - started
    report = dict(tables=payload['manifest']['tables'], elapsed_seconds=elapsed,
                  rto_target_seconds=p['rto_target_seconds'], rto_met=elapsed <= p['rto_target_seconds'],
                  snapshot_age_seconds=age, refund_effects=effects,
                  policy='Tenant 1 original policy', limitations=LIMITATIONS)
    (HOME / 'drill-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['snapshot', 'drill'])
    parser.add_argument('--offline-demo', action='store_true')
    parser.add_argument('--target')
    args = parser.parse_args()
    if args.operation == 'snapshot':
        snapshot(args.offline_demo)
    else:
        drill(args.target)


if __name__ == '__main__':
    main()
