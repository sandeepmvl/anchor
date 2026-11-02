from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / 'config/fine_tuning.yml'


def load_policy(path=POLICY_PATH):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def score(feature, policy=None):
    policy = load_policy() if policy is None else policy
    methods = policy['methods']
    for key in ('name', 'evidence'):
        if not isinstance(feature.get(key), str) or not feature[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    if type(feature.get('residual_failure')) is not bool:
        raise ValueError('residual_failure must be a boolean')
    volatility = feature.get('knowledge_volatility', 1)
    low, high = policy['knowledge_volatility_range']
    if type(volatility) is not int or not low <= volatility <= high:
        raise ValueError(f'knowledge_volatility must be an integer from {low} to {high}')
    axes = {}
    totals = dict.fromkeys(methods, 0)
    for axis, options in policy['axes'].items():
        value = feature.get(axis)
        if not isinstance(value, str) or value not in options:
            raise ValueError(f'{axis}: choose from {list(options)}')
        weight = volatility if axis == 'freshness' else 1
        points = dict(zip(methods, (p * weight for p in options[value])))
        axes[axis] = {'value': value, 'weight': weight, 'points': points}
        for method, points_value in points.items():
            totals[method] += points_value
    gates = {
        'stable facts': feature['freshness'] == 'stable',
        'behavioral target': feature['target'] in ('format', 'tone'),
        'approved labels and separate evaluation cases': feature['labels'] == 'ready',
        'funded maintenance': feature['maintenance'] == 'funded',
        'failure remains after prompt/schema fixes': feature['residual_failure'],
    }
    blocked = [name for name, passed in gates.items() if not passed]
    eligible = [m for m in policy['tie_order'] if m != 'fine-tune' or not blocked]
    winner = max(eligible, key=totals.get)
    if winner == 'RAG':
        explanation = ('Retrieve current tenant-scoped policy evidence for this ticket; '
                       'changing policy facts belong in retrieval, not memorized weights.')
    elif winner == 'fine-tune':
        explanation = ('Gated experiment only: verify approved examples and separated held-out '
                       'cases, then compare quality, latency, total cost, and correction burden.')
    else:
        explanation = 'Retain the prompt/schema baseline; training is not needed for this decision.'
    return {
        'policy_version': policy['version'],
        'name': feature['name'],
        'evidence': feature['evidence'],
        'axes': axes,
        'totals': totals,
        'fine_tune_eligible': not blocked,
        'fine_tune_blockers': blocked,
        'recommendation': winner,
        'experiment_required': winner == 'fine-tune',
        'explanation': explanation,
        'advisory_only': True,
        'refund_authorized': False,
        'model_override': None,
    }
