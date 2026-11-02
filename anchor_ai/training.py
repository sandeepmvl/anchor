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


# Recorded projection shapes: (input width, output width, occurrences).
# Real training must compare this inventory with the loaded model's modules.
LORA_INVENTORY = {
    'q_proj': (896, 896, 24),
    'k_proj': (896, 128, 24),
    'v_proj': (896, 128, 24),
    'o_proj': (896, 896, 24),
    'up_proj': (896, 4864, 24),
    'gate_proj': (896, 4864, 24),
    'down_proj': (4864, 896, 24),
}


def validate_adapter(spec):
    request = spec.get('adapter')
    if request is None:
        return None
    if spec['base_model'] != 'Qwen/Qwen2.5-0.5B-Instruct':
        raise ValueError('unsupported_adapter_inventory')
    rank = request.get('rank')
    alpha = request.get('alpha')
    targets = request.get('target_modules')
    if type(rank) is not int or rank < 1:
        raise ValueError('invalid_adapter_rank')
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('invalid_adapter_alpha')
    if (not isinstance(targets, list) or not targets
            or any(not isinstance(t, str) or t not in LORA_INVENTORY for t in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError('invalid_adapter_targets')
    modules = []
    for name in sorted(targets):
        inputs, outputs, count = LORA_INVENTORY[name]
        if rank > min(inputs, outputs):
            raise ValueError('invalid_adapter_rank')
        modules.append({'target': name, 'input': inputs, 'output': outputs,
                        'occurrences': count,
                        'trainable_parameters': count * rank * (inputs + outputs)})
    parameters = sum(m['trainable_parameters'] for m in modules)
    budget = request.get('max_trainable_parameters')
    if type(budget) is not int or budget < 1 or parameters > budget:
        raise ValueError('adapter_parameter_budget_exceeded')
    base_count = spec['settings']['parameter_count']
    return {'method': 'lora', 'rank': rank, 'alpha': alpha,
            'scaling_rule': 'alpha/r', 'effective_scale': alpha / rank,
            'target_modules': sorted(targets), 'modules': modules,
            'trainable_parameters': parameters,
            'max_trainable_parameters': budget,
            'trainable_percent_of_declared_base': 100 * parameters / base_count,
            'base_parameter_count': base_count, 'base_weights_frozen': True,
            'bias': 'none', 'dropout': 0.0, 'use_rslora': False,
            'inventory_source': 'recorded_shapes_not_loaded_model_inspection',
            'evidence_status': 'configuration_only_not_adapter_training',
            'total_training_memory_measured': False}


def lora_forward(weight, a, b, x, alpha):
    import numpy as np
    if (weight.ndim != 2 or a.ndim != 2 or b.ndim != 2
            or a.shape[0] < 1 or a.shape[1] != weight.shape[1]
            or b.shape != (weight.shape[0], a.shape[0])):
        raise ValueError('invalid_adapter_shapes')
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('invalid_adapter_alpha')
    return np.matmul(weight, x) + (alpha / a.shape[0]) * np.matmul(b, np.matmul(a, x))


def estimate_training_memory(parameters, adapter_parameters, mode,
                             activation_bytes, runtime_overhead_bytes,
                             budget_bytes, double_quant=True):
    """Planning model, not allocator measurements; all quantities are bytes."""
    values = (parameters, adapter_parameters, activation_bytes,
              runtime_overhead_bytes, budget_bytes)
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError('invalid_memory_assumption')
    if mode not in ('fp16', '8-bit', 'NF4') or type(double_quant) is not bool:
        raise ValueError('invalid_quantization_mode')
    blocks = math.ceil(parameters / 64)
    metadata = 0
    if mode == '8-bit':
        metadata = blocks * 4
    elif mode == 'NF4':
        metadata = blocks + math.ceil(blocks / 256) * 4 if double_quant else blocks * 4
    components = {
        'base_weights': math.ceil(parameters * {'fp16': 2, '8-bit': 1, 'NF4': 0.5}[mode]),
        'quantization_metadata': metadata,
        'adapters': adapter_parameters * 2,
        'gradients': adapter_parameters * 2,
        'optimizer_state': adapter_parameters * 8,
        'activation_allowance': activation_bytes,
        'runtime_overhead': runtime_overhead_bytes,
    }
    total = sum(components.values())
    accepted = total <= budget_bytes
    return {'mode': mode, 'components_bytes': components,
            'estimated_total_bytes': total, 'budget_bytes': budget_bytes,
            'accepted': accepted,
            'decision': 'estimated_feasible' if accepted else 'rejected_memory_budget',
            'warning': None if accepted else 'Estimated total exceeds selected hardware budget',
            'evidence_status': 'estimate_not_gpu_run',
            'training_launched': False, 'checkpoint_created': False,
            'assumptions': '2-byte adapters/gradients; two fp32 AdamW moments; no base gradients; '
                           'illustrative blockwise int8, not LLM.int8 outlier accounting; '
                           'allowances must be validated against measured peaks'}


def quantized_training_preflight(spec, adapter=None):
    request = spec.get('quantized_training')
    if request is None:
        return None
    adapter = validate_adapter(spec) if adapter is None else adapter
    if adapter is None:
        raise ValueError('quantized_training_requires_adapter')
    bnb = request['bitsandbytes']
    if (bnb['load_in_4bit'] is not True or bnb['bnb_4bit_quant_type'] != 'nf4'
            or type(bnb['bnb_4bit_use_double_quant']) is not bool
            or bnb['bnb_4bit_compute_dtype'] not in ('bfloat16', 'float32', 'float16')):
        raise ValueError('invalid_nf4_configuration')
    comparisons = {
        mode: estimate_training_memory(
            spec['settings']['parameter_count'], adapter['trainable_parameters'], mode,
            request['activation_allowance_bytes'], request['runtime_overhead_bytes'],
            request['gpu_budget_bytes'], bnb['bnb_4bit_use_double_quant'])
        for mode in ('fp16', '8-bit', 'NF4')
    }
    return {**comparisons['NF4'], 'comparisons': comparisons,
            'bitsandbytes': dict(bnb), 'adapter': adapter,
            'hardware': request['hardware'],
            'precision_requirement': 'Validate compute and reduction support on the selected GPU; '
                                     'this offline check does not probe hardware',
            'training_options': request['training_options']}


def experiment(policy=None):
    policy = load_policy() if policy is None else policy
    spec = policy['sft']
    adapter = validate_adapter(spec)
    preflight = quantized_training_preflight(spec, adapter)
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
    if adapter is not None:
        contract.update(adapter=adapter)
    if preflight is not None:
        contract.update(proposed_quantized_training=spec['quantized_training'])
    settings = spec['settings']
    result = {'candidate': spec['candidate'], 'experiment_id': digest(contract),
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
    if preflight is not None:
        result['qlora_preflight'] = preflight
    return result


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


def recorded_adapter_evaluation(candidate='adapter-recorded', manifest_path=None):
    from anchor_ai.adapters import MANIFEST, evaluate_release
    return evaluate_release(candidate, MANIFEST if manifest_path is None else manifest_path)
