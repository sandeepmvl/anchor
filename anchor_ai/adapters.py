import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'config/model_releases.yml'
SOURCE = 'constructed_recording_not_measured'
NF4 = np.array([-1., -.6961928, -.5250731, -.3949175, -.2844414,
                -.1847734, -.09105, 0., .0795803, .1609302, .2461123,
                .3379152, .4407098, .562617, .7229568, 1.], dtype=np.float32)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def recording(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 1 or rows[0].get('source') != SOURCE:
        raise ValueError('invalid_recording')
    return rows[0]


@dataclass(frozen=True)
class AdapterSlice:
    weight: np.ndarray
    a: np.ndarray
    b: np.ndarray
    scale: float

    def forward(self, x):
        return self.weight @ x + self.scale * (self.b @ (self.a @ x))

    def merged(self):
        return self.weight + self.scale * (self.b @ self.a)


def load_slice(directory, base_model, base_revision, config=None):
    directory = Path(directory)
    config = read_json(directory / 'adapter_config.json') if config is None else dict(config)
    if (config.get('base_model_name_or_path') != base_model
            or config.get('revision') != base_revision):
        raise ValueError('adapter_base_identity_mismatch')
    r, alpha = config.get('r'), config.get('lora_alpha')
    if type(r) is not int or r < 1:
        raise ValueError('invalid_adapter_rank')
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('invalid_adapter_alpha')
    if config.get('use_rslora') is not False or config.get('target_modules') != ['q_proj']:
        raise ValueError('unsupported_slice_configuration')
    arrays = [np.asarray(read_json(directory / name), dtype=np.float32)
              for name in ('base_weight.json', 'a.json', 'b.json')]
    w, a, b = arrays
    if (any(v.ndim != 2 or not np.isfinite(v).all() for v in arrays)
            or a.shape != (r, w.shape[1]) or b.shape != (w.shape[0], r)):
        raise ValueError('invalid_adapter_shapes')
    # Freeze both the arrays and the scale captured at load time.
    for array in arrays:
        array.setflags(write=False)
    return AdapterSlice(w, a, b, alpha / r)


def nf4_roundtrip(weight):
    scale = float(np.max(np.abs(weight)))
    if scale == 0:
        return np.zeros_like(weight)
    indices = np.abs((weight / scale)[..., None] - NF4).argmin(axis=-1)
    return NF4[indices] * scale


def paired_comparison(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if (reference.shape != (50,) or candidate.shape != reference.shape
            or not np.isin(reference, [0, 1]).all()
            or not np.isin(candidate, [0, 1]).all()):
        raise ValueError('invalid_paired_scores')
    differences = candidate.astype(float) - reference.astype(float)
    indices = np.random.default_rng(2026).integers(0, 50, size=(10000, 50))
    low, high = np.quantile(differences[indices].mean(axis=1), [.025, .975])
    return {'baseline': float(reference.mean()), 'candidate': float(candidate.mean()),
            'difference': float(differences.mean()), 'paired_95_ci': [float(low), float(high)],
            'regression': bool(high < 0)}


def comparisons(record):
    scores = {key: [int(c) for c in ''.join(value.split())]
              for key, value in record['scores'].items()}
    if scores['wrong_alpha'] != scores['wrong_alpha_unmerged']:
        raise ValueError('wrong_alpha_self_comparison_must_agree')
    return {name: paired_comparison(scores['unmerged_correct'], scores[name])
            for name in ('fp_reload', 'wrong_alpha', 'nf4_merge')}


def precision_evidence(artifact_kind, evidence):
    if artifact_kind not in ('floating', 'quantized'):
        raise ValueError('unknown_artifact_kind')
    if artifact_kind == 'quantized' and (
            evidence.get('artifact_kind') != 'quantized'
            or not evidence.get('artifact_hash') or evidence.get('passed') is not True):
        raise ValueError('separate_quantized_artifact_evidence_required')


def artifact_contract(manifest_path=MANIFEST):
    path = Path(manifest_path)
    manifest = yaml.safe_load(path.read_text())
    if manifest.get('version') != 1:
        raise ValueError('unsupported_release_manifest')
    expected = manifest['expected']
    actual = {name: read_json(ROOT / manifest['fixture'] / name) for name in expected}
    replies = recording(ROOT / manifest['replies'])
    wanted = {name: digest(value) for name, value in expected.items()}
    wanted['heldout_replies'] = digest(manifest['expected_replies'])
    found = {name: digest(value) for name, value in actual.items()}
    found['heldout_replies'] = digest(replies)
    if found != wanted:
        raise ValueError('release_artifact_hash_mismatch')
    base = manifest['base']
    load_slice(ROOT / manifest['fixture'], base['model'], base['revision'])
    return manifest, replies, found


def evaluate_release(candidate, manifest_path=MANIFEST):
    manifest, record, hashes = artifact_contract(manifest_path)
    if candidate not in manifest['releases']:
        raise ValueError('unknown_candidate')
    release = manifest['releases'][candidate]
    results = comparisons(record)
    scenario = release['scenario']
    gates = record['recorded_gates']
    checks = {
        'offline_only': release.get('offline_only') is True,
        'reply_available': candidate in record['replies'],
        'no_demonstrated_merge_regression': not results[scenario]['regression'],
        'recorded_golden_dataset': gates.get('golden_dataset') is True,
        'recorded_llm_judge': gates.get('llm_judge') is True,
        'recorded_retrieval_regression': gates.get('retrieval_regression') is True,
        'no_critical_failures': gates.get('critical_failures') == 0,
    }
    # The nf4 recording diagnoses rounding; it is not evidence for an exported artifact.
    if scenario == 'nf4_merge':
        checks['separate_quantized_artifact_evidence'] = False
    blockers = [name for name, passed in checks.items() if not passed]
    return {'candidate': candidate, 'source': SOURCE, 'artifact_hashes': hashes,
            'manifest_hash': digest(manifest), 'comparisons': results,
            'checks': checks, 'blockers': blockers, 'offline_approved': not blockers,
            'deployment_approved': False, 'refund_authorized': False}


def numeric_report(directory):
    config = read_json(Path(directory) / 'adapter_config.json')
    good = load_slice(directory, config['base_model_name_or_path'], config['revision'])
    wrong_config = dict(config, lora_alpha=config['lora_alpha'] / 4)
    wrong = load_slice(directory, config['base_model_name_or_path'], config['revision'], wrong_config)
    x = np.eye(good.weight.shape[1], dtype=np.float32)
    correct_error = float(np.max(np.abs(good.merged() @ x - good.forward(x))))
    wrong_error = float(np.max(np.abs(wrong.merged() @ x - wrong.forward(x))))
    base_nf4 = nf4_roundtrip(good.weight)
    effective = base_nf4 + good.scale * (good.b @ good.a)
    return {'W': good.weight.tolist(), 'A': good.a.tolist(), 'B': good.b.tolist(),
            'scale_fixed_at_load': good.scale,
            'floating_merge_output_error': correct_error,
            'wrong_alpha_self_output_error': wrong_error,
            'wrong_vs_correct_weight_error': float(np.max(np.abs(wrong.merged() - good.merged()))),
            'base_nf4_error': float(np.max(np.abs(base_nf4 - good.weight))),
            'additional_nf4_merge_error': float(np.max(np.abs(nf4_roundtrip(effective) - effective))),
            'warning': 'one-block illustrative NF4; not bitsandbytes or a model quality evaluation'}
