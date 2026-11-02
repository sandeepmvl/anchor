from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.customization import load_policy
from anchor_ai.dataset_quality import write_rows, unique_words, source_registry


def prepare():
    policy = load_policy()
    paths = [Path(p) for p in policy['synthetic']['source_paths']]
    if not paths[0].exists():
        if any(path.exists() for path in paths[1:]):
            raise ValueError('Partial source exports exist; prepare a consistent export first')
        for path, split, count in zip(paths, ('train', 'validation', 'held_out'), (100, 1, 1)):
            rows = []
            for i in range(count):
                rows.append({'id': f'demo-{split}-{i}', 'tenant_id': 'demo',
                    'source': 'real', 'split': split, 'training_permitted': True,
                    'educational_fixture': True,
                    'input': 'Reviewed refund request ' + unique_words(f'{split}-{i}'),
                    'output': policy['synthetic']['approved_outputs']['refund_review'],
                    'label': 'refund_review'})
            write_rows(path, rows)
        print('Created educational exports; source=real means curated side of this fixture only')
    else:
        print('Preserving existing training export and its provenance')
    registry = source_registry(policy)
    print('Source registry rows:', len(registry))


if __name__ == '__main__':
    prepare()
