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


def sft_examples(rows, policy=None):
    policy = load_policy() if policy is None else policy
    result = []
    for row in rows:
        if set(row) != {'input', 'output', 'label'}:
            raise ValueError('invalid_curated_schema')
        if row['label'] not in policy['dataset']['labels']:
            raise ValueError('invalid_curated_label')
        if any(not isinstance(row[k], str) or not row[k].strip()
               for k in ('input', 'output')):
            raise ValueError('empty_training_text')
        result.append({'prompt': [{'role': 'user', 'content': row['input']}],
                       'completion': [{'role': 'assistant', 'content': row['output']}]})
    return result


def synthetic_batch_decision(generated, rejected, real, existing_synthetic,
                             accepted, policy=None):
    policy = load_policy() if policy is None else policy
    limits = policy['synthetic']
    counts = (generated, rejected, real, existing_synthetic, accepted)
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('invalid_batch_counts')
    if generated != rejected + accepted:
        raise ValueError('invalid_batch_counts')
    rate = rejected / generated if generated else None
    ratio = (existing_synthetic + accepted) / real if real else None
    low, high = limits['reject_rate_band']
    within = (rate is not None and ratio is not None
              and low <= rate <= high
              and ratio <= limits['max_synthetic_real_ratio'])
    return {'reject_rate': rate, 'synthetic_real_ratio': ratio,
            'within_bands': within,
            'decision': 'eligible_for_review' if within else 'hold',
            'merged': False, 'training_release': False}


def sft_activation_decision(manifest, evaluation=None, retrieval=None, checkpoint=None):
    """Consume trusted internal evaluator results, never a customer request body."""
    evaluation = evaluation or {}
    retrieval = retrieval or {}
    selected = manifest['selection']['selected_checkpoint']
    checkpoint = selected if checkpoint is None else checkpoint
    checks = {
        'validation_selected_checkpoint': checkpoint == selected,
        'training_release': manifest.get('training_release') is True,
        'measured_experiment': manifest.get('evidence_status') == 'measured',
        'matching_dataset_lineage': evaluation.get('lineage') == manifest['lineage'],
        'matching_experiment': evaluation.get('experiment_id') == manifest['experiment_id'],
        'matching_candidate': evaluation.get('candidate') == manifest['candidate'],
        'matching_checkpoint': evaluation.get('checkpoint') == checkpoint,
        'held_out_evidence': (
            evaluation.get('heldout_digest') == manifest['lineage']['held_out']
            and bool(evaluation.get('reply_digest'))),
        'measured_held_out_evidence': evaluation.get('source') == 'measured',
        'held_out_quality_and_safety': evaluation.get('passed') is True,
        'existing_retrieval_regression_gate': (
            retrieval.get('gate') == 'existing_retrieval_regression'
            and retrieval.get('passed') is True
            and retrieval.get('candidate') == manifest['candidate']
            and retrieval.get('experiment_id') == manifest['experiment_id']
            and retrieval.get('checkpoint') == checkpoint
            and bool(retrieval.get('artifact_digest'))),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    active = not blockers
    return {'candidate': manifest['candidate'], 'requested_checkpoint': checkpoint,
            'selected_checkpoint': selected, 'evidence_status': manifest['evidence_status'],
            'active': active, 'blockers': blockers,
            'model_override': checkpoint if active else None,
            'refund_authorized': False}


def distillation_decision(teacher, student, route, spec):
    gaps = {label: teacher['per_class'][label]['f1-score']
            - student['per_class'][label]['f1-score'] for label in spec['labels']}
    weak = [label for label, gap in gaps.items() if gap > spec['max_f1_gap']]
    economical = (student['estimated_usd_per_ticket'] < teacher['usd_per_ticket']
                  and student['latency']['p95_ms'] < teacher['latency']['p95_ms'])
    decision = 'accept' if not weak and economical else ('route' if route['passes'] else 'reject')
    return {'decision': decision, 'weak_classes': weak, 'f1_gaps': gaps,
            'economical': economical, 'driving_figure': (
                {'largest_f1_gap': max(gaps.values())} if weak else
                {'student_p95_ms': student['latency']['p95_ms'],
                 'estimated_usd_per_ticket': student['estimated_usd_per_ticket']}),
            'advisory_only': True, 'deployment_approved': False,
            'refund_authorized': False, 'model_override': None}
