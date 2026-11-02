import argparse
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.multi_agent_judge import ReleaseRejected, judge, require


ROOT = Path(__file__).resolve().parents[1]


def load_adapter():
    name = os.environ.get('ANCHOR_ORCHESTRATION_RELEASE_ADAPTER')
    require(bool(name), 'product_composition_adapter_missing')
    module = importlib.import_module(name)
    factory = getattr(module, 'build_release_adapter', None)
    require(callable(factory), 'release_adapter_factory_missing')
    adapter = factory()
    for method in ('preflight', 'execute_case', 'human_only_drill', 'rollback_drill'):
        require(callable(getattr(adapter, method, None)),
                'release_adapter_missing_' + method)
    return adapter


def check_preflight(adapter):
    result = adapter.preflight()
    require(isinstance(result, dict), 'invalid_preflight')
    for key in ('queue', 'fake_model', 'seeded_retrieval', 'framework',
                'local_mcp', 'sandbox_image', 'sandbox_isolation'):
        require(result.get(key) is True, 'preflight_failed_' + key)
    return result


def run_gate(adapter, recordings):
    check_preflight(adapter)
    cases = [json.loads(line) for line in
             (ROOT / 'eval/datasets/multi_agent_cases.jsonl').read_text().splitlines()
             if line.strip()]
    for case in cases:
        evidence = adapter.execute_case(case)
        require(not inspect.isawaitable(evidence), 'adapter_must_finish_execution')
        judge(case, evidence, recordings)
        print('PASS multi-agent case:', case['id'])
    print('PASS multi-agent execution evidence and recorded judge')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--human-only-drill', action='store_true')
    parser.add_argument('--rollback-drill', action='store_true')
    args = parser.parse_args()
    adapter = load_adapter()
    if args.preflight:
        check_preflight(adapter)
        print('PASS local orchestration dependency preflight')
    elif args.human_only_drill:
        result = adapter.human_only_drill()
        require(result.get('full_redacted_handoff') is True, 'human_handoff_failed')
        require(result.get('no_post_switch_dispatch') is True, 'switch_not_enforced')
        print('PASS automation stopped; full redacted handoff retained')
    elif args.rollback_drill:
        result = adapter.rollback_drill()
        require(result.get('atomic') is True, 'rollback_not_atomic')
        require(result.get('prior_version_restored') is True, 'rollback_failed')
        require(result.get('app_deployment_required') is False, 'rollback_needs_deploy')
        print('PASS prior orchestration version restored without app deployment')
    else:
        run_gate(adapter, os.environ.get('ANCHOR_MULTI_AGENT_JUDGE_RECORDINGS'))


if __name__ == '__main__':
    try:
        main()
    except (ReleaseRejected, ImportError, OSError, ValueError) as error:
        print('BLOCKED multi-agent release:', type(error).__name__, str(error))
        raise SystemExit(1)
