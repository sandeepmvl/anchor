import base64
import copy
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from ops.backups import verify_snapshot as recovery


class BackupIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.home = Path(self.directory.name)
        self.env = patch.dict(os.environ, ANCHOR_BACKUP_KEY='ab' * 32, RAILS_ENV='test')
        self.env.start()
        self.addCleanup(self.env.stop)
        self.location = patch.object(recovery, 'HOME', self.home)
        self.location.start()
        self.addCleanup(self.location.stop)
        self.source = self.home / 'source.sqlite3'
        recovery.fixture(self.source)
        self.payload = recovery.package(self.source)

    def test_encryption_authenticates_before_plaintext(self):
        encrypted = recovery.crypt(recovery.canonical(self.payload), 'encrypt')
        self.assertNotIn(b'Tenant 1 original policy', encrypted)
        self.assertEqual(json.loads(recovery.crypt(encrypted, 'decrypt')), self.payload)
        corrupted = encrypted[:-1] + bytes([encrypted[-1] ^ 1])
        with self.assertRaises(ValueError):
            recovery.crypt(corrupted, 'decrypt')
        with patch.dict(os.environ, ANCHOR_BACKUP_KEY='cd' * 32):
            with self.assertRaises(ValueError):
                recovery.crypt(encrypted, 'decrypt')

    def test_missing_tampered_and_incompatible_index(self):
        missing = copy.deepcopy(self.payload)
        del missing['artifacts']['index.json']
        with self.assertRaisesRegex(ValueError, 'missing_or_unknown'):
            recovery.verify(missing)
        tampered = copy.deepcopy(self.payload)
        tampered['artifacts']['index.json']['generations'][0]['dimensions'] = 99
        with self.assertRaisesRegex(ValueError, 'tampered_artifact'):
            recovery.verify(tampered)
        tampered['manifest']['artifact_sha256']['index.json'] = recovery.sha(
            recovery.canonical(tampered['artifacts']['index.json']))
        with self.assertRaisesRegex(ValueError, 'incompatible_index'):
            recovery.verify(tampered)
        self.assertFalse((self.home / 'disposable.sqlite3').exists())

    def test_record_checksums_are_independent_of_file_hash(self):
        with sqlite3.connect(self.source) as db:
            db.execute("UPDATE recovery_memory SET value='changed' WHERE tenant_id=1")
        raw = self.source.read_bytes()
        self.payload['database'] = base64.b64encode(raw).decode()
        self.payload['manifest']['database_sha256'] = recovery.sha(raw)
        with self.assertRaisesRegex(ValueError, 'record_checksum_mismatch'):
            recovery.verify(self.payload)

    def test_production_and_non_disposable_targets_refused(self):
        with self.assertRaisesRegex(ValueError, 'production_coordinated'):
            recovery.snapshot()
        production = self.home / 'production.sqlite3'
        production.write_text('untouched')
        with self.assertRaisesRegex(ValueError, 'disposable_target'):
            recovery.drill(production)
        self.assertEqual(production.read_text(), 'untouched')
        with patch.dict(os.environ, RAILS_ENV='production'):
            with self.assertRaisesRegex(ValueError, 'production_coordinated'):
                recovery.snapshot(True)
            with self.assertRaisesRegex(ValueError, 'production_target'):
                recovery.drill()

    def test_retention_keeps_hourly_and_daily_points(self):
        now = 2000000000
        for hours in (0, 1, 25, 49, 200):
            path = self.home / f'snapshot-{hours}.enc'
            path.write_bytes(b'fixture')
            os.utime(path, (now - hours * 3600, now - hours * 3600))
        recovery.retain(self.home, now)
        self.assertTrue((self.home / 'snapshot-0.enc').exists())
        self.assertTrue((self.home / 'snapshot-1.enc').exists())
        self.assertTrue((self.home / 'snapshot-49.enc').exists())
        self.assertFalse((self.home / 'snapshot-200.enc').exists())

    def test_isolated_restore_contract(self):
        recovery.snapshot(True)
        report = recovery.drill()
        self.assertEqual(report['refund_effects'], 1)
        self.assertTrue(report['rto_met'])
        with recovery.connection(self.home / 'disposable.sqlite3') as db:
            self.assertEqual(recovery.recovery_inventory(db), report['tables'])
            for tenant in (1, 2):
                _, rows = recovery.retrieval_snapshot(db, tenant)
                self.assertEqual([r['text'] for r in rows], [f'Tenant {tenant} original policy'])
                self.assertEqual(rows[0]['checksum'], recovery.sha(rows[0]['text'].encode()))
                self.assertEqual(db.execute('SELECT result FROM recovery_actions WHERE tenant_id=?', (tenant,)).fetchone()[0], 'completed')
                self.assertEqual(db.execute('SELECT value FROM recovery_memory WHERE tenant_id=?', (tenant,)).fetchone()[0], 'refund completed')
                self.assertEqual(db.execute('SELECT ticket FROM recovery_conversations WHERE tenant_id=?', (tenant,)).fetchone()[0], 'refund-ticket')
            recovery.verify(recovery.package(self.home / 'disposable.sqlite3'))


if __name__ == '__main__':
    unittest.main()
