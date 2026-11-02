import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest
from anchor_ai.customization import load_policy
from anchor_ai.dataset_quality import audit_distillation, read_rows
from anchor_ai.distillation import evaluate
from anchor_ai.query_analyzer import QueryAnalyzer
from anchor_ai.training import distillation_manifest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'eval/finetune/fixtures/teacher_student_predictions.jsonl'


@pytest.fixture(scope='module')
def run():
    rows = read_rows(FIXTURE)
    report, student = evaluate(rows)
    return rows, report, student


def test_manifest_and_independent_reports(run):
    rows, report, student = run
    contract = report['manifest']['contract']
    assert contract['teacher_tasks'] == ['intent', 'tone']
    assert contract['student_label2id']['intent'] == {'refund': 0, 'dispute': 1}
    assert contract['student_id2label']['tone'] == {'0': 'neutral', '1': 'frustrated'}
    assert report['manifest']['excluded_tasks'] == ['policy_answering', 'tool_authorization', 'generation']
    assert not report['deployment_approved']
    held = [r for r in rows if r['split'] == 'held_out']
    assert report['teacher_shared_call']['usd_total'] == sum(r['teacher_usd'] for r in held)
    for task in ('intent', 'tone'):
        value = report['tasks'][task]
        teacher, learner = value['teacher'], value['student']
        assert teacher['ticket_ids'] == learner['ticket_ids'] == [r['id'] for r in held]
        assert set(learner['per_class']) == set(student.spec['labels'][task])
        assert sum(map(sum, learner['confusion']['counts'])) == len(held)
        assert 0 <= learner['macro_f1'] <= 1
        assert learner['training']['wall_seconds'] > 0
        assert learner['training']['estimated_compute_usd'] > 0
        assert learner['estimated_usd_per_ticket'] > 0
        assert learner['latency']['p95_ms'] >= learner['latency']['p50_ms'] > 0
        assert teacher['evidence_status'] == 'constructed_recording_not_measured'
        assert 'driving_figure' in value['decision']
    intent = report['tasks']['intent']
    assert intent['teacher']['accuracy'] == 1
    assert intent['student']['per_class']['dispute']['recall'] == 0
    assert set(intent['route']['fallback_classes']) == {'refund', 'dispute'}
    assert intent['route']['fallback_tickets'] == len(held)
    assert intent['route']['quality']['predictions'] == intent['teacher']['predictions']
    assert intent['route']['quality']['macro_f1'] == intent['teacher']['macro_f1']
    assert intent['route']['estimated_usd_per_ticket'] == pytest.approx(
        intent['teacher']['usd_per_ticket'] + intent['student']['estimated_usd_per_ticket'])
    assert intent['route']['estimated_usd_per_ticket'] > intent['teacher']['usd_per_ticket']
    assert not intent['route']['passes']
    assert intent['decision']['decision'] == 'reject'
    assert report['tasks']['tone']['teacher']['accuracy'] < 1
    assert report['tasks']['tone']['student']['accuracy'] == 1


def test_audit_rejects_leakage_and_wrong_supervision(run):
    rows, _, _ = run
    spec = load_policy()['distillation']
    broken = deepcopy(rows)
    broken[-1]['family'] = broken[0]['family']
    with pytest.raises(ValueError, match='cross_split_family'):
        audit_distillation(broken, spec)
    broken = deepcopy(rows)
    broken[-1]['input'] = broken[0]['input']
    with pytest.raises(ValueError, match='cross_split_exact_duplicate'):
        audit_distillation(broken, spec)
    broken = deepcopy(rows)
    broken[-1]['label_source'] = 'reviewed_teacher'
    with pytest.raises(ValueError, match='incorrect_label_source'):
        audit_distillation(broken, spec)
    assert distillation_manifest(rows)['experiment_id'] != distillation_manifest(rows, balanced=True)['experiment_id']


def test_query_hook_preserves_default_and_boundaries(run):
    _, _, student = run

    class Client:
        def __init__(self):
            self.calls = []

        async def complete(self, text):
            self.calls.append(text)
            return json.dumps({'queries': [text]})

    async def check():
        client = Client()
        analyzer = QueryAnalyzer(client, classifier=student)
        baseline = await QueryAnalyzer(Client()).analyze('Please refund chargeback.')
        assert (await analyzer.analyze('Please refund chargeback.')).model_dump() == baseline.model_dump()
        low = await analyzer.analyze_with_classification('Please refund chargeback.', 'tone', 'demo')
        assert low['classification']['reason'] == 'low_confidence'
        assert low['fallback_target'] == 'existing_query_analyzer'
        for task in ('policy_answering', 'tool_authorization'):
            result = await analyzer.analyze_with_classification('Please refund chargeback.', task, 'demo')
            assert result['classification']['reason'] == 'unsupported_task'
            assert result['policy_answer'] is None
            assert not result['tool_authorized'] and not result['refund_authorized']
        unknown = await analyzer.analyze_with_classification('quasar nebula zephyr', 'intent', 'demo')
        assert unknown['classification']['reason'] == 'out_of_scope_vocabulary'
        assert len(client.calls) == 5
        assert student.classify('Please refund return.', 'intent', 'other')['reason'] == 'tenant_boundary'
        from copy import deepcopy
        accepted_student = deepcopy(student)
        accepted_student.spec['confidence_threshold'] = 0
        accepted_student.weak_classes['tone'] = []
        accepted = await QueryAnalyzer(client, classifier=accepted_student).analyze_with_classification(
            'Please refund return.', 'tone', 'demo')
        assert not accepted['classification']['fallback']
        assert accepted['plan']['queries'] == ['Please refund return.']
        assert not accepted['tool_authorized'] and accepted['policy_answer'] is None
    asyncio.run(check())
