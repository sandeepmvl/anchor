import hashlib
import json
import math
from pathlib import Path

from anchor_ai.customization import load_policy

ROOT = Path(__file__).resolve().parents[1]
RECORDED = 'constructed_recording_not_measured'


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def analyze_loss(rows):
    if not rows or [r['epoch'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('expected_consecutive_epochs')
    for row in rows:
        for key in ('train_loss', 'val_loss'):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid_loss')
    pairs = list(zip(rows, rows[1:]))
    train_falls = bool(pairs) and all(b['train_loss'] < a['train_loss'] for a, b in pairs)
    val_falls = bool(pairs) and all(b['val_loss'] < a['val_loss'] for a, b in pairs)
    upturns = [b['epoch'] for a, b in pairs if b['val_loss'] > a['val_loss']]
    best = min(rows, key=lambda r: (r['val_loss'], r['epoch']))
    if train_falls and val_falls:
        status = 'still_falling_short_run_inconclusive'
    elif train_falls and upturns:
        status = 'overfitting_after_validation_minimum'
    else:
        status = 'inspect_plateau_and_split_audit'
    return {'best_epoch': best['epoch'], 'best_val_loss': best['val_loss'],
            'selected_checkpoint': f"checkpoint-epoch{best['epoch']}",
            'final_epoch': rows[-1]['epoch'],
            'first_upturn': upturns[0] if upturns else None,
            'interpretation': status, 'loss_digest': digest(rows)}


def experiment(policy=None):
    policy = load_policy() if policy is None else policy
    spec = policy['sft']
    rows = spec['recorded_rows']
    ids = {row['id']: row for row in rows}
    if len(ids) != len(rows):
        raise ValueError('duplicate_id')
    families, texts = {}, {}
    for row in rows:
        split = row['split']
        if split not in ('train', 'validation', 'held_out'):
            raise ValueError('invalid_split')
        family = row['family']
        if families.setdefault(family, split) != split:
            raise ValueError('cross_split_family')
        text = ' '.join(row['input'].lower().split())
        if texts.setdefault(text, split) != split:
            raise ValueError('cross_split_exact_duplicate')
        if not row['input'].strip() or not row['output'].strip():
            raise ValueError('empty_example')
        if row['source'] == 'synthetic':
            seed = ids.get(row.get('seed_id'))
            if (row['status'] != 'recorded_accepted' or not seed
                    or seed['source'] != 'curated'
                    or (seed['family'], seed['split']) != (family, split)):
                raise ValueError('unaccepted_or_unlinked_synthetic')
        elif row['source'] != 'curated' or row['status'] != 'recorded_approved':
            raise ValueError('unapproved_source')
    splits = {split: [r for r in rows if r['split'] == split]
              for split in ('train', 'validation', 'held_out')}
    if any(not values for values in splits.values()):
        raise ValueError('empty_split')
    losses = read_jsonl(ROOT / spec['loss_log'])
    if any(r.get('source') != RECORDED for r in losses):
        raise ValueError('recording_source_mismatch')
    selection = analyze_loss(losses)
    lineage = {split: digest(values) for split, values in splits.items()}
    contract = {key: spec[key] for key in
                ('base_model', 'base_revision', 'tokenizer_revision',
                 'chat_template', 'settings', 'acceptance')}
    contract.update(lineage=lineage, loss_digest=selection['loss_digest'])
    settings = spec['settings']
    return {'candidate': spec['candidate'], 'experiment_id': digest(contract),
            'evidence_status': RECORDED, 'lineage': lineage,
            'counts': {key: len(value) for key, value in splits.items()},
            'train_sources': {source: sum(r['source'] == source for r in splits['train'])
                              for source in ('curated', 'synthetic')},
            'contract': contract, 'selection': selection,
            'parameter_state_bytes': settings['parameter_count'] * settings['bytes_per_parameter'],
            'gpu_budget_bytes': settings['gpu_budget_bytes'],
            'activation_budget_is_additional': True,
            'tokenized': False, 'training_launched': False,
            'checkpoint_created': False, 'training_release': False,
            'split_audit': 'family_and_exact_checks_only; near-duplicate_review_required'}


def recorded_evaluation(manifest, policy=None):
    policy = load_policy() if policy is None else policy
    spec = policy['sft']
    replies = read_jsonl(ROOT / spec['heldout_replies'])
    heldout = {r['id']: r for r in spec['recorded_rows'] if r['split'] == 'held_out'}
    if len(replies) != len(heldout) or {r['id'] for r in replies} != set(heldout):
        raise ValueError('heldout_coverage_mismatch')
    selected = manifest['selection']['selected_checkpoint']
    if any(r['checkpoint'] != selected or r['source'] != RECORDED for r in replies):
        raise ValueError('reply_checkpoint_or_source_mismatch')
    for row in replies:
        if (type(row['critical_failure']) is not bool
                or row['candidate_quality'] not in (0, 1)
                or row['base_quality'] not in (0, 1)):
            raise ValueError('invalid_recorded_review')
    quality = sum(r['candidate_quality'] for r in replies) / len(replies)
    base_quality = sum(r['base_quality'] for r in replies) / len(replies)
    failures = sum(r['critical_failure'] for r in replies)
    limits = spec['acceptance']
    return {'candidate': manifest['candidate'], 'experiment_id': manifest['experiment_id'],
            'lineage': manifest['lineage'], 'checkpoint': selected,
            'source': RECORDED, 'heldout_digest': manifest['lineage']['held_out'],
            'reply_digest': digest(replies), 'quality': quality,
            'base_quality': base_quality, 'critical_failures': failures,
            'passed': quality >= limits['min_quality'] and quality >= base_quality
                      and failures <= limits['max_critical_failures'],
            'paired_replies': [{**row, 'input': heldout[row['id']]['input']}
                               for row in replies]}


def distillation_manifest(rows, policy=None, balanced=False):
    from anchor_ai.dataset_quality import audit_distillation
    policy = load_policy() if policy is None else policy
    spec = policy['distillation']
    splits = audit_distillation(rows, spec)
    mappings = {task: {label: i for i, label in enumerate(labels)}
                for task, labels in spec['labels'].items()}
    contract = {'teacher': spec['teacher'], 'rubric': spec['rubric_version'],
                'teacher_tasks': list(mappings), 'student_label2id': mappings,
                'student_id2label': {t: {str(i): label for label, i in m.items()}
                                     for t, m in mappings.items()},
                'student': 'TF-IDF + logistic regression, one pipeline per task',
                'class_weight': 'balanced' if balanced else None,
                'confidence_threshold': spec['confidence_threshold'],
                'max_f1_gap': spec['max_f1_gap'],
                'lineage': {s: digest(values) for s, values in splits.items()}}
    return {'candidate': spec['candidate'], 'experiment_id': digest(contract),
            'contract': contract, 'evidence_status': RECORDED,
            'counts': {s: len(values) for s, values in splits.items()},
            'supervision': 'reviewed teacher hard labels only; independent human evaluation',
            'excluded_tasks': ['policy_answering', 'tool_authorization', 'generation'],
            'training_release': False, 'deployment_approved': False,
            'split_audit': 'family and exact checks; human near-duplicate review still required'}
