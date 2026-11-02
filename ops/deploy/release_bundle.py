import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIELDS = ('contract', 'index', 'model', 'offline_only', 'prompt', 'reply', 'ticket')


def identity(bundle):
    if set(bundle) != set(FIELDS):
        raise ValueError('invalid_bundle_fields')
    if bundle['offline_only'] is not True or any(
            not isinstance(bundle[k], str) or not bundle[k] for k in FIELDS if k != 'offline_only'):
        raise ValueError('offline_fixture_bundle_required')
    canonical = {k: bundle[k] for k in FIELDS}
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class PinnedBundle:
    tenant_id: int
    generation: int
    bundle_id: str
    prompt: str
    model: str
    index: str
    contract: str
    ticket: str
    reply: str

    @property
    def cache_namespace(self):
        return f'{self.tenant_id}:{self.bundle_id}'

    def complete(self, ticket):
        if ticket != self.ticket:
            raise ValueError('release_replay_miss')
        return self.reply


def pin(snapshot, *, tenant_id, required_contract, offline=False):
    # Copy before validating; caller mutations cannot alter admitted work.
    snapshot = json.loads(json.dumps(snapshot, allow_nan=False))
    if type(tenant_id) is not int or tenant_id <= 0 or snapshot['tenant_id'] != tenant_id:
        raise ValueError('release_tenant_mismatch')
    if type(snapshot['generation']) is not int or snapshot['generation'] < 0:
        raise ValueError('invalid_release_generation')
    bundle = snapshot['bundle']
    if identity(bundle) != snapshot['bundle_id']:
        raise ValueError('bundle_identity_changed')
    if bundle['contract'] != required_contract:
        raise ValueError('incompatible_bundle_roll_forward_required')
    if not offline:
        raise ValueError('production_readiness_evidence_required')
    return PinnedBundle(tenant_id, snapshot['generation'], snapshot['bundle_id'],
                        bundle['prompt'], bundle['model'], bundle['index'],
                        bundle['contract'], bundle['ticket'], bundle['reply'])


def main():
    parser = argparse.ArgumentParser(description='Offline release-bundle boundary')
    parser.add_argument('--gate', required=True)
    args = parser.parse_args()
    from eval.release_gate import release_bundle_fixture_gate
    target = json.loads(args.gate)
    identity(target)
    print(json.dumps(release_bundle_fixture_gate(target)))


if __name__ == '__main__':
    main()
