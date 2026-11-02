import argparse
import json
import sqlite3
from pathlib import Path


class ConfigurationConflict(ValueError):
    pass


class OrchestrationStore:
    """Use one shared durable database; do not use a worker-local database."""
    def __init__(self, path):
        self.path = str(path)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS versions (
                    version TEXT PRIMARY KEY, configuration TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS active (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    version TEXT NOT NULL REFERENCES versions(version),
                    human_only INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS handoffs (
                    tenant_id INTEGER NOT NULL, run_id TEXT NOT NULL,
                    context TEXT NOT NULL,
                    PRIMARY KEY(tenant_id, run_id));
                CREATE TABLE IF NOT EXISTS audit (
                    sequence INTEGER PRIMARY KEY, operator TEXT NOT NULL,
                    operation TEXT NOT NULL, version TEXT NOT NULL);
            ''')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=2)
        db.execute('PRAGMA foreign_keys=ON')
        return db

    def register(self, version, configuration):
        if not isinstance(version, str) or not version.strip():
            raise ValueError('invalid_version')
        if not isinstance(configuration, dict):
            raise ValueError('invalid_configuration')
        encoded = json.dumps(configuration, sort_keys=True, allow_nan=False)
        with self.connect() as db:
            previous = db.execute('SELECT configuration FROM versions WHERE version=?',
                                  (version,)).fetchone()
            if previous and previous[0] != encoded:
                raise ConfigurationConflict('immutable_version')
            db.execute('INSERT OR IGNORE INTO versions VALUES (?, ?)', (version, encoded))
            db.execute('INSERT OR IGNORE INTO active VALUES (1, ?, 1)', (version,))

    def snapshot(self):
        with self.connect() as db:
            row = db.execute('''SELECT active.version, human_only, configuration
                FROM active JOIN versions USING(version) WHERE singleton=1''').fetchone()
        if row is None:
            raise ValueError('orchestration_not_initialized')
        return dict(version=row[0], human_only=bool(row[1]), configuration=json.loads(row[2]))

    def activate(self, expected, target, operator):
        if not operator:
            raise ValueError('operator_required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM versions WHERE version=?', (target,)).fetchone() is None:
                raise ValueError('unknown_version')
            changed = db.execute('UPDATE active SET version=? WHERE singleton=1 AND version=?',
                                 (target, expected)).rowcount
            if changed != 1:
                raise ConfigurationConflict('active_version_changed')
            db.execute('INSERT INTO audit(operator,operation,version) VALUES (?, ?, ?)',
                       (operator, 'activate', target))

    def human_only(self, enabled, operator):
        if type(enabled) is not bool or not operator:
            raise ValueError('invalid_mode_change')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT version FROM active WHERE singleton=1').fetchone()
            if row is None:
                raise ValueError('orchestration_not_initialized')
            db.execute('UPDATE active SET human_only=? WHERE singleton=1', (int(enabled),))
            db.execute('INSERT INTO audit(operator,operation,version) VALUES (?, ?, ?)',
                       (operator, 'human_only' if enabled else 'automated', row[0]))

    def save_handoff(self, tenant_id, run_id, context):
        required = {'ticket', 'evidence', 'completed_actions', 'pending_actions',
                    'approval_state', 'trace_id', 'orchestration_version', 'reason'}
        if (type(tenant_id) is not int or tenant_id <= 0 or not run_id
                or not required.issubset(context)):
            raise ValueError('incomplete_handoff')
        if not self.snapshot()['human_only']:
            raise ValueError('human_only_required')
        # Redact every string, including nested evidence and action results.
        from anchor_ai.pii import prepare
        def redact(value):
            if isinstance(value, str):
                return prepare(value)
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            return value
        safe = redact(context)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            mode = db.execute('SELECT human_only FROM active WHERE singleton=1').fetchone()
            if not mode or not mode[0]:
                raise ConfigurationConflict('mode_changed')
            db.execute('INSERT OR REPLACE INTO handoffs VALUES (?, ?, ?)',
                       (tenant_id, run_id, json.dumps(safe)))
        return safe

    def handoff(self, tenant_id, run_id):
        with self.connect() as db:
            row = db.execute('SELECT context FROM handoffs WHERE tenant_id=? AND run_id=?',
                             (tenant_id, run_id)).fetchone()
        return json.loads(row[0]) if row else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', required=True)
    parser.add_argument('--operator', required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('stop')
    rollback = commands.add_parser('rollback')
    rollback.add_argument('--expected', required=True)
    rollback.add_argument('--target', required=True)
    args = parser.parse_args()
    if not Path(args.database).is_file():
        parser.error('shared orchestration database does not exist')
    store = OrchestrationStore(args.database)
    if args.command == 'stop':
        store.human_only(True, args.operator)
    else:
        if not store.snapshot()['human_only']:
            parser.error('activate human-only mode before rollback')
        store.activate(args.expected, args.target, args.operator)
    print(json.dumps(store.snapshot(), sort_keys=True))


if __name__ == '__main__':
    main()
