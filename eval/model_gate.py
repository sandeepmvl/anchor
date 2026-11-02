import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.customization import sft_activation_decision
from anchor_ai.training import experiment, recorded_evaluation


# Preference alignment lesson 3688
def replay_dpo():
    from anchor_ai.customization import load_policy, dpo_candidate_decision
    from anchor_ai.dataset_quality import read_rows
    from anchor_ai.preferences import ROOT, RECORDED, analyze
    from anchor_ai.training import dpo_experiment, digest
    policy = load_policy()
    spec = policy['dpo']
    manifest = dpo_experiment(policy)
    logs = read_rows(ROOT / spec['logs'])
    replies = read_rows(ROOT / spec['replies'])
    if any(r['policy_id'] != spec['policy']['id']
           or r['reference_id'] != spec['reference']['id']
           or r['beta'] != spec['beta'] for r in logs):
        raise ValueError('log_manifest_identity_mismatch')
    report = analyze(logs, replies, manifest['heldout_ids'])
    cases = read_rows(ROOT / spec['safety_cases'])
    if (not cases or len({r['id'] for r in cases}) != len(cases)
            or any(r['candidate'] != manifest['candidate']
                   or r['source'] != RECORDED
                   or type(r['passed']) is not bool for r in cases)):
        raise ValueError('invalid_safety_recording')
    required = ('grounding', 'privacy', 'injection', 'escalation', 'tool_safety')
    checks = {}
    for name in required:
        group = [r for r in cases if r['check'] == name]
        checks[name] = bool(group) and all(r['passed'] for r in group)
    safety = {'candidate': manifest['candidate'],
              'experiment_id': manifest['experiment_id'], 'source': RECORDED,
              'checks': checks, 'case_digest': digest(cases)}
    decision = dpo_candidate_decision(manifest, report, safety)
    return {'manifest': manifest, 'evaluation': report, 'safety': safety,
            'decision': decision, 'deployment_approved': False,
            'required_existing_gates': ['golden_dataset_and_versioned_judge',
                'tenant_document_and_customer_message_red_team',
                'unit_tests', 'retrieval_regressions', 'agent_trajectories', 'Rails_CI'],
            'retrieval_evidence': 'existing_gates_still_required_not_replaced',
            'demo': 'Same refund ticket: choose justified human handoff over unauthorized confident resolution.'}


def replay(candidate):
    if candidate == 'prompt-regression-recorded':
        from eval.regression_gate import gate
        path = Path(__file__).resolve().parents[1] / 'tmp/regression_report.json'
        try:
            evidence = json.loads(path.read_text())
            decision = gate(evidence, release=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            decision = {'passed': False, 'blockers': ['missing_or_invalid_regression_evidence'],
                        'error': str(error)}
        return {'decision': decision, 'deployment_approved': False,
                'retrieval_evidence': 'existing_gates_still_required_not_replaced'}
    if candidate == 'dpo-recorded':
        return replay_dpo()
    if candidate in ('adapter-recorded', 'default-recorded'):
        from anchor_ai.training import recorded_adapter_evaluation
        evidence = recorded_adapter_evaluation(candidate)
        return {'evaluation': evidence,
                'decision': {'active': evidence['offline_approved'],
                             'scope': 'offline_recordings_only',
                             'blockers': evidence['blockers'],
                             'model_override': candidate if evidence['offline_approved'] else None,
                             'refund_authorized': False},
                'retrieval_evidence': 'constructed_recording_not_measured',
                'deployment_approved': False}
    if candidate == 'distilled-labels':
        from anchor_ai.dataset_quality import read_rows
        from anchor_ai.distillation import evaluate
        path = Path(__file__).resolve().parent / 'finetune/fixtures/teacher_student_predictions.jsonl'
        report, _ = evaluate(read_rows(path))
        return {'evaluation': report,
                'decision': {'active': False, 'model_override': None,
                             'refund_authorized': False,
                             'blockers': ['constructed_teacher_evidence',
                                          'no_locked_release_evaluation',
                                          'golden_dataset_and_judge_regressions_required',
                                          'retrieval_and_agent_regressions_required']},
                'retrieval_evidence': 'not_run_for_a_trained_candidate',
                'deployment_approved': False}
    manifest = experiment()
    if candidate != manifest['candidate']:
        raise ValueError('unknown_candidate')
    evidence = recorded_evaluation(manifest)
    decision = sft_activation_decision(manifest, evidence)
    return {'evaluation': evidence, 'decision': decision,
            'retrieval_evidence': 'not_run_for_a_trained_candidate',
            'deployment_approved': decision['active']}


def main():
    parser = argparse.ArgumentParser(description='Replay recorded evidence, not a deployment gate pass.')
    parser.add_argument('--offline', action='store_true', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--artifact')
    parser.add_argument('--baseline-artifact')
    parser.add_argument('--artifact-report')
    parser.add_argument('--runtime', default='vllm_gpu')
    parser.add_argument('--expect-rejected', action='store_true')
    args = parser.parse_args()
    if args.artifact:
        from anchor_ai.serving_artifacts import selection
        try:
            report = json.loads(Path(args.artifact_report).read_text()) if args.artifact_report else None
            decision = selection(args.artifact, args.runtime, report, args.baseline_artifact)
        except (OSError, ValueError, KeyError, TypeError):
            decision = {'selectable': False, 'deployment_approved': False,
                        'blockers': ['missing_or_invalid_artifact_evidence']}
        print(json.dumps({'decision': decision}, indent=2, sort_keys=True))
        if args.expect_rejected:
            if decision['selectable']:
                raise SystemExit('expected rejection')
        elif not decision['selectable']:
            raise SystemExit(1)
        return
    if args.expect_rejected or args.artifact_report or args.baseline_artifact:
        parser.error('artifact options require --artifact')
    print(json.dumps(replay(args.candidate), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
