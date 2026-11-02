import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.adapters import MANIFEST, ROOT, artifact_contract, digest, evaluate_release

STATE = ROOT / '.git/anchor-model-release.json'


@contextmanager
def selection_lock(path):
    # Deployment commands run on the Linux release host, not inside a web request.
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
            name = handle.name
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def approved_entry(candidate, manifest_path):
    report = evaluate_release(candidate, manifest_path)
    if report['offline_approved'] is not True:
        raise ValueError('release_gate_failed:' + ','.join(report['blockers']))
    return {'candidate': candidate, 'artifact_hashes': report['artifact_hashes'],
            'manifest_hash': report['manifest_hash'], 'offline_only': True}


def current_state(path=STATE, manifest_path=MANIFEST):
    path = Path(path)
    if path.exists():
        value = json.loads(path.read_text())
        if value.get('version') != 1 or value.get('scope') != 'offline':
            raise ValueError('invalid_release_state')
        return value
    manifest, _, _ = artifact_contract(manifest_path)
    return {'version': 1, 'scope': 'offline', 'generation': 0,
            'current': approved_entry(manifest['default'], manifest_path), 'previous': None}


def transition(candidate=None, rollback=False, path=STATE, manifest_path=MANIFEST):
    if rollback == (candidate is not None):
        raise ValueError('choose_promote_or_rollback')
    with selection_lock(path):
        state = current_state(path, manifest_path)
        old = state['current']
        # Fail closed if the active selection's pinned identities no longer match.
        if approved_entry(old['candidate'], manifest_path) != old:
            raise ValueError('active_release_identity_changed')
        if rollback:
            if state['previous'] is None:
                raise ValueError('no_previous_release')
            target = state['previous']
            if approved_entry(target['candidate'], manifest_path) != target:
                raise ValueError('rollback_release_identity_changed')
        else:
            target = approved_entry(candidate, manifest_path)
        if target == old:
            return state
        result = dict(state, generation=state['generation'] + 1,
                      current=target, previous=old)
        atomic_write(path, result)
        return result


@dataclass(frozen=True)
class PinnedRelease:
    candidate: str
    generation: int
    artifact_identity: str
    ticket: str
    reply: str

    def complete(self, ticket):
        if ticket != self.ticket:
            raise ValueError('release_replay_miss')
        return self.reply


def pin_release(path=STATE, manifest_path=MANIFEST):
    state = current_state(path, manifest_path)
    selected = state['current']
    manifest, record, hashes = artifact_contract(manifest_path)
    if (selected.get('offline_only') is not True
            or selected['artifact_hashes'] != hashes
            or selected['manifest_hash'] != digest(manifest)):
        raise ValueError('selected_release_identity_mismatch')
    candidate = selected['candidate']
    if candidate not in manifest['releases']:
        raise ValueError('unknown_selected_release')
    return PinnedRelease(candidate, state['generation'], digest(hashes),
                         record['ticket'], record['replies'][candidate])


def main():
    parser = argparse.ArgumentParser(description='Offline-only model release selection')
    parser.add_argument('--offline', required=True, action='store_true')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--promote')
    group.add_argument('--rollback', action='store_true')
    parser.add_argument('--state', type=Path, default=STATE)
    args = parser.parse_args()
    before = pin_release(args.state)
    state = transition(args.promote, args.rollback, args.state)
    after = pin_release(args.state)
    ticket = before.ticket
    print(json.dumps({'scope': 'offline', 'same_ticket': ticket,
                      'in_flight_release': before.candidate,
                      'in_flight_reply': before.complete(ticket),
                      'subsequent_release': after.candidate,
                      'subsequent_reply': after.complete(ticket),
                      'generation': state['generation'],
                      'artifact_identity': after.artifact_identity,
                      'deployment_approved': False}, indent=2))


if __name__ == '__main__':
    main()
