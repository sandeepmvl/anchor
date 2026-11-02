import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.customization import sft_activation_decision
from anchor_ai.training import experiment, recorded_evaluation


def replay(candidate):
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
    args = parser.parse_args()
    print(json.dumps(replay(args.candidate), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
