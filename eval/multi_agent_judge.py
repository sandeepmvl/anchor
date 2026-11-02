import hashlib
import json
from pathlib import Path


class ReleaseRejected(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise ReleaseRejected(reason)


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_evidence(case, evidence):
    require(isinstance(evidence, dict), 'missing_execution_evidence')
    require(evidence.get('case_id') == case['id'], 'case_mismatch')
    require(evidence.get('tenant_id') == case['tenant_id'], 'tenant_mismatch')
    require(evidence.get('ticket_ref') == case['ticket_ref'], 'ticket_mismatch')
    require(evidence.get('async_job', {}).get('performed') is True,
            'ticket_not_processed_asynchronously')
    require(bool(evidence.get('async_job', {}).get('job_id')), 'missing_job_identity')
    require(evidence.get('unapproved_mutations') == 0, 'unapproved_mutation')
    trace_id = evidence.get('trace_id')
    require(isinstance(trace_id, str) and len(trace_id) == 32, 'missing_trace')
    spans = evidence.get('spans', [])
    require(bool(spans), 'missing_spans')
    require(all(s.get('trace_id') == trace_id for s in spans), 'broken_trace')
    require(all('raw_message' not in s and 'stdout' not in s and 'script' not in s
                for s in spans), 'unsafe_trace_payload')
    if case['mode'] == 'human_only':
        require(evidence.get('automation_stopped') is True, 'automation_not_stopped')
        handoff = evidence.get('human_handoff', {})
        require(handoff.get('persisted') is True, 'handoff_not_persisted')
        require(handoff.get('redacted') is True, 'handoff_not_redacted')
        require(handoff.get('tenant_id') == case['tenant_id'], 'handoff_scope')
        for key in ('ticket', 'evidence', 'completed_actions', 'pending_actions',
                    'approval_state', 'trace_id', 'orchestration_version', 'reason'):
            require(key in handoff, 'handoff_missing_' + key)
        require(handoff['trace_id'] == trace_id, 'handoff_trace')
        require(bool(handoff['reason']), 'handoff_reason')
        require(evidence.get('post_switch_tool_calls') == 0,
                'tool_dispatched_after_switch')
        return
    citations = evidence.get('citations', [])
    require(bool(citations), 'missing_citations')
    for citation in citations:
        require(citation.get('resolved') is True, 'unresolved_citation')
        require(citation.get('seeded') is True, 'unseeded_citation')
        require(citation.get('tenant_id') == case['tenant_id'], 'citation_scope')
        require(bool(citation.get('chunk_id')), 'missing_chunk_identity')
        require(bool(citation.get('content_sha256')), 'missing_chunk_digest')
    handoff = evidence.get('specialist_handoff', {})
    require(handoff.get('persisted') is True, 'specialist_handoff_not_durable')
    require(handoff.get('resumed_after_restart') is True,
            'specialist_restart_not_exercised')
    require(handoff.get('context_digest_before') ==
            handoff.get('context_digest_after') and
            bool(handoff.get('context_digest_before')), 'specialist_context_lost')
    require(bool(handoff.get('from_role')) and bool(handoff.get('to_role')) and
            handoff['from_role'] != handoff['to_role'], 'missing_specialization')
    mcp = evidence.get('mcp', {})
    require(mcp.get('transport') in ('stdio', 'streamable_http'), 'fixture_only_mcp')
    require(mcp.get('local_server') is True, 'local_mcp_not_exercised')
    require(all(mcp.get(k) is True for k in
                ('initialized', 'discovered', 'called', 'authorized')),
            'incomplete_mcp_execution')
    sandbox = evidence.get('sandbox', {})
    require(sandbox.get('isolated') is True, 'missing_sandbox_isolation')
    require(sandbox.get('service_execution') is True, 'sandbox_service_not_called')
    require(bool(sandbox.get('execution_id')), 'missing_sandbox_execution_id')
    require(sandbox.get('approved_script') is True, 'unapproved_script')
    require(sandbox.get('exit_reason') == 'completed', 'sandbox_did_not_complete')
    require(sandbox.get('egress_denied') is True, 'sandbox_egress_not_enforced')
    require(sandbox.get('limits_enforced') is True, 'sandbox_limits_not_enforced')
    require(set(('retrieval', 'llm', 'mcp', 'sandbox', 'handoff')).issubset(
        {s.get('stage') for s in spans}), 'missing_integration_span')


def recorded_judgment(case, evidence, recording_path):
    require(recording_path is not None, 'missing_judge_recordings')
    records = [json.loads(line) for line in Path(recording_path).read_text().splitlines()
               if line.strip()]
    matches = [r for r in records if r.get('case_id') == case['id'] and
               r.get('input_sha256') == fingerprint(evidence)]
    require(len(matches) == 1, 'judge_replay_miss')
    record = matches[0]
    require(bool(record.get('judge_model')), 'missing_judge_provenance')
    require(bool(record.get('rubric_version')), 'missing_judge_rubric')
    require(record.get('grounded') is True, 'judge_groundedness_regression')
    require(record.get('handoff_complete') is True, 'judge_handoff_regression')
    score = record.get('score')
    require(type(score) in (int, float) and 0.9 <= score <= 1,
            'judge_score_regression')
    return record


def judge(case, evidence, recording_path):
    validate_evidence(case, evidence)
    return recorded_judgment(case, evidence, recording_path)
