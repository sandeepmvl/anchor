import hashlib
import json
import math
import time
from pathlib import Path

import httpx
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'eval/datasets/self_hosted_quality_cases.jsonl'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('manifest_must_be_object')
    if (value.get('format') not in {'fp16', 'gptq', 'awq', 'gguf'}
            or type(value.get('simulated')) is not bool
            or type(value.get('bits')) is not int
            or value['bits'] not in {4, 8, 16}):
        raise ValueError('invalid_artifact_manifest')
    return value


def policy():
    return yaml.safe_load((ROOT / 'config/self_hosted.yml').read_text())['artifacts']


def sizes(path):
    manifest = load(path)
    # Never interpret the manifest itself as model weights.
    return {'manifest_bytes': Path(path).stat().st_size,
            'weight_bytes': None,
            'weight_size_basis': 'no_verified_weights',
            'estimated_weight_bytes': manifest.get('estimated_weight_bytes'),
            'estimate_basis': 'simulated_payload_excludes_metadata_and_runtime'}


def numerical_experiment(bits):
    # A real tiny round-to-nearest experiment, NOT GPTQ, AWQ, or K-quants.
    weights = np.linspace(-1.0, 1.0, 257)
    levels = 2 ** bits - 1
    restored = np.round((weights + 1) * levels / 2) * 2 / levels - 1
    return {'method': 'generated_round_to_nearest_not_format_quantization',
            'bits': bits, 'values': len(weights),
            'mse': float(np.mean((weights - restored) ** 2)),
            'max_absolute_error': float(np.max(np.abs(weights - restored)))}


def fixture_transport(manifest, cases):
    by_prompt = {case['prompt']: case for case in cases}

    def dispatch(request):
        if request.method != 'POST' or request.url.path != '/v1/chat/completions':
            return httpx.Response(404)
        payload = json.loads(request.content)
        if (payload.get('model') != 'tiny-llama'
                or payload.get('temperature') != 0
                or payload.get('max_tokens') != 32
                or payload.get('stream') is not False):
            return httpx.Response(400)
        messages = payload.get('messages', [])
        if len(messages) != 1 or messages[0].get('role') != 'user':
            return httpx.Response(400)
        case = by_prompt.get(messages[0].get('content'))
        if case is None:
            return httpx.Response(400)
        reply = dict(case['expected'])
        if manifest.get('fixture_scenario') == 'refund_regression' and case['id'] == 'refund':
            reply['dry_run'] = False
        return httpx.Response(200, json={
            'id': 'offline-fixture', 'object': 'chat.completion',
            'model': 'tiny-llama', 'choices': [{'index': 0,
                'message': {'role': 'assistant', 'content': json.dumps(reply)},
                'finish_reason': 'stop'}]})

    return httpx.MockTransport(dispatch)


def evaluate(manifest, cases):
    rows = []
    start = time.perf_counter()
    with httpx.Client(transport=fixture_transport(manifest, cases), trust_env=False) as client:
        for case in cases:
            before = time.perf_counter()
            response = client.post('http://offline/v1/chat/completions', json={
                'model': 'tiny-llama', 'messages': [{'role': 'user', 'content': case['prompt']}],
                'temperature': 0, 'max_tokens': 32, 'stream': False})
            response.raise_for_status()
            reply = json.loads(response.json()['choices'][0]['message']['content'])
            rows.append({'id': case['id'], 'passed': reply == case['expected'],
                         'latency_ms': (time.perf_counter() - before) * 1000})
    elapsed = time.perf_counter() - start
    return {'rows': rows, 'pass_rate': sum(r['passed'] for r in rows) / len(rows),
            'completed_req_s': len(rows) / elapsed,
            'completion_latency_mean_ms': sum(r['latency_ms'] for r in rows) / len(rows)}


def compare(baseline, candidate, runtime):
    base, cand = load(baseline), load(candidate)
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('invalid_quality_cases')
    before, after = evaluate(base, cases), evaluate(cand, cases)
    return {'baseline_sha256': digest(baseline), 'candidate_sha256': digest(candidate),
            'dataset_sha256': digest(CASES), 'runtime': runtime,
            'latency_mode': 'measured_in_process_fixture_not_model_inference',
            'quality_basis': 'fixture_exact_match_not_llm_judge',
            'sizes': {'baseline': sizes(baseline), 'candidate': sizes(candidate)},
            'baseline': before, 'candidate': after,
            'quality_loss': before['pass_rate'] - after['pass_rate'],
            'numerical_experiment': numerical_experiment(cand['bits']),
            'warning': 'No model weights, GPU, vLLM, real quantization, or production quality verified.'}


def selection(candidate, runtime, report=None, baseline=None):
    blockers = []
    try:
        manifest = load(candidate)
        rules = policy()
        if runtime not in rules['compatible_runtimes'].get(manifest['format'], []):
            blockers.append('incompatible_runtime')
        if manifest['simulated']:
            blockers.append('simulated_manifest_not_quantized_weights')
        # This adapter deliberately has no trusted hardware-evidence importer.
        # A JSON field claiming measured_real_runtime cannot enable production.
        blockers.extend(['missing_verified_weight_files',
                         'trusted_measured_runtime_evidence_required',
                         'golden_dataset_and_versioned_judge_required'])
        valid = (isinstance(report, dict) and baseline is not None
                 and report.get('candidate_sha256') == digest(candidate)
                 and report.get('baseline_sha256') == digest(baseline)
                 and report.get('dataset_sha256') == digest(CASES)
                 and report.get('runtime') == runtime
                 and report.get('quality_basis') == 'fixture_exact_match_not_llm_judge'
                 and report.get('latency_mode') == 'measured_in_process_fixture_not_model_inference')
        if valid:
            # Recompute deterministic quality, not timing, to reject edited scores.
            expected = compare(baseline, candidate, runtime)
            for key in ('baseline', 'candidate'):
                if (report[key]['rows'] and
                        [(r['id'], r['passed']) for r in report[key]['rows']] !=
                        [(r['id'], r['passed']) for r in expected[key]['rows']]):
                    valid = False
                score = report[key]['pass_rate']
                if (type(score) not in (float, int) or not math.isfinite(score)
                        or score != expected[key]['pass_rate']):
                    valid = False
            loss = report['quality_loss']
            if (type(loss) not in (float, int) or not math.isfinite(loss)
                    or loss != expected['quality_loss']):
                valid = False
            if (expected['candidate']['pass_rate'] < rules['min_pass_rate']
                    or expected['quality_loss'] > rules['max_quality_loss']):
                blockers.append('quality_threshold_failed')
        if not valid:
            blockers.append('missing_or_invalid_comparison')
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        blockers.append('missing_or_invalid_artifact_evidence')
    return {'selectable': False, 'deployment_approved': False,
            'blockers': sorted(set(blockers)),
            'scope': 'offline_artifact_policy_only'}
