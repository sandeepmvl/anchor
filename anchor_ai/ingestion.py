import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid
from .pii import CONFIG, ROOT, prepare
from .storage import connect, persist


def sentence_nodes(text):
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+', text) if part.strip()]
    size = CONFIG['sentences_per_parent']
    radius = CONFIG['sentence_window']
    nodes = []
    for start in range(0, len(sentences), size):
        group = sentences[start:start + size]
        parent = len(nodes)
        nodes.append(dict(chunk_index=parent, parent_index=None, is_leaf=False, text=' '.join(group)))
        for offset, sentence in enumerate(group):
            position = start + offset
            nodes.append(dict(chunk_index=len(nodes), parent_index=parent, is_leaf=True,
                text=sentence, window_text=' '.join(sentences[max(0, position-radius):position+radius+1])))
    return nodes


def stage_path(reference):
    if not re.fullmatch(r'[0-9a-f]{32}', reference):
        raise ValueError('invalid_upload_reference')
    root = Path(os.environ.get('ANCHOR_STAGE_ROOT', str(ROOT / 'storage/ingestion_uploads')))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root / (reference + '.json')


def stage(tenant_id, raw, suffix='.txt'):
    if type(tenant_id) is not int or tenant_id <= 0:
        raise ValueError('invalid_tenant')
    text = prepare(raw, suffix)
    reference = uuid.uuid4().hex
    path = stage_path(reference)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as target:
        json.dump(dict(tenant_id=tenant_id, text=text), target)
    return reference


def ingest(tenant_id, reference, db_path):
    staged = json.loads(stage_path(reference).read_text())
    if type(tenant_id) is not int or tenant_id <= 0 or staged['tenant_id'] != tenant_id:
        raise ValueError('upload_tenant_mismatch')
    text = prepare(staged['text'])
    content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    from .app import chunk_embed
    from .schemas import ChunkEmbedRequest
    db = connect(db_path)
    try:
        return persist(db, tenant_id, reference, text, content_hash,
                       lambda: chunk_embed(ChunkEmbedRequest(tenant_id=tenant_id, text=text)))
    finally:
        db.close()


def sync_directory(db, tenant_id, directory, client=None, chunker_version='sentence-v1'):
    from .index_maintenance import maintain_document, delete_document, valid_tenant
    valid_tenant(tenant_id)
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError('source_inventory_unavailable')
    scope = str(directory)
    inventory = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {'.md', '.html', '.txt'} or not path.is_file():
            continue
        text = prepare(path.read_text(encoding='utf-8'), path.suffix.lower())
        if not text.strip():
            raise ValueError('empty_source')
        sidecar = path.with_name(path.name + '.metadata.json')
        metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {'title': path.stem}
        if not isinstance(metadata, dict):
            raise ValueError('invalid_metadata')
        inventory[path.name] = (text, metadata)
    result = []
    for external_id, (text, metadata) in inventory.items():
        result.append(maintain_document(db, tenant_id, external_id, text, metadata,
                      chunker_version, scope, client=client))
    vanished = db.execute('SELECT external_id FROM documents WHERE tenant_id=? AND source_scope=?',
                          (tenant_id, scope)).fetchall()
    for row in vanished:
        if row['external_id'] not in inventory:
            delete_document(db, tenant_id, row['external_id'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'ingest'])
    parser.add_argument('--tenant', type=int, required=True)
    parser.add_argument('--reference')
    parser.add_argument('--suffix', default='.txt')
    args = parser.parse_args()
    try:
        if args.action == 'stage':
            result = {'reference': stage(args.tenant, sys.stdin.read(), args.suffix)}
        else:
            result = ingest(args.tenant, args.reference, os.environ['ANCHOR_DB_PATH'])
        print(json.dumps(result))
    except Exception:
        print('ingestion_failed', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
