import math
from pathlib import Path

from anchor_ai.dataset_quality import sanitize, has_pii, read_rows

RECORDED = 'constructed_recording_not_measured'
ROOT = Path(__file__).resolve().parents[1]


def build_pairs(rows, tenant):
    if not tenant or not rows:
        raise ValueError('tenant_and_transcripts_required')
    result = []
    for original in rows:
        if original.get('tenant_id') != tenant:
            continue
        if original.get('training_permitted') is not True:
            raise ValueError('training_not_permitted')
        row = sanitize(original)
        review = row.get('review', {})
        if (review.get('approved') is not True or not review.get('reviewer_id')
                or not str(review.get('rationale', '')).strip()
                or review.get('basis') != 'grounding_authorization_and_helpfulness'):
            raise ValueError('explicit_review_required_not_outcome_labels')
        replies = row.get('replies', [])
        if len(replies) != 2 or len({r['id'] for r in replies}) != 2:
            raise ValueError('two_distinct_replies_required')
        for reply in replies:
            if (reply.get('tenant_id') != tenant
                    or reply.get('context_id') != row.get('context_id')
                    or not str(reply.get('text', '')).strip()):
                raise ValueError('reply_tenant_or_context_mismatch')
        chosen_id = review.get('chosen_id')
        if chosen_id not in {r['id'] for r in replies}:
            raise ValueError('review_must_identify_chosen_reply')
        chosen = next(r for r in replies if r['id'] == chosen_id)
        rejected = next(r for r in replies if r['id'] != chosen_id)
        if chosen['text'] == rejected['text']:
            raise ValueError('identical_completions')
        if not row.get('customer') or not row.get('context') or not row.get('context_id'):
            raise ValueError('shared_prompt_context_required')
        pair = {
            'id': row['id'], 'tenant_id': tenant, 'family': row['family'],
            'split': row['split'], 'context_id': row['context_id'],
            'prompt': row['context'] + '\nCustomer: ' + row['customer'],
            'chosen': chosen['text'], 'rejected': rejected['text'],
            'review': review, 'training_permitted': True,
            'source': row['source'],
            'pii_redacted': sanitize(original) != original,
            'dispositions': {'chosen': chosen['disposition'],
                             'rejected': rejected['disposition']}}
        if has_pii(pair):
            raise ValueError('redaction_failed')
        result.append(pair)
    audit_pairs(result)
    return result


def audit_pairs(rows):
    if not rows or len({r['tenant_id'] for r in rows}) != 1:
        raise ValueError('one_nonempty_tenant_dataset_required')
    splits = {s: [] for s in ('train', 'validation', 'held_out')}
    ids, families, prompts = set(), {}, {}
    for row in rows:
        split = row['split']
        if row['id'] in ids or split not in splits:
            raise ValueError('duplicate_id_or_invalid_split')
        ids.add(row['id'])
        prompt = ' '.join(row['prompt'].lower().split())
        if families.setdefault(row['family'], split) != split:
            raise ValueError('cross_split_family')
        if prompts.setdefault(prompt, split) != split:
            raise ValueError('cross_split_exact_duplicate')
        if has_pii(row) or row.get('training_permitted') is not True:
            raise ValueError('unsanitized_or_unpermitted_pair')
        if not row['review']['rationale'].strip():
            raise ValueError('review_rationale_required')
        splits[split].append(row)
    if any(not group for group in splits.values()):
        raise ValueError('empty_split')
    return splits


def trainer_fields(rows):
    return [{k: r[k] for k in ('prompt', 'chosen', 'rejected')} for r in rows]


def analyze(logs, replies, heldout_ids=None):
    if not logs or not replies:
        raise ValueError('logs_and_replies_required')
    summaries = []
    expected_ids = set(heldout_ids) if heldout_ids is not None else None
    previous_step = -1
    identities = None
    for row in logs:
        if row.get('source') != RECORDED or row.get('split') != 'held_out':
            raise ValueError('offline_heldout_evidence_required')
        if type(row['step']) is not int or row['step'] <= previous_step:
            raise ValueError('steps_must_increase')
        previous_step = row['step']
        beta = row['beta']
        if type(beta) not in (int, float) or not math.isfinite(beta) or beta <= 0:
            raise ValueError('invalid_beta')
        identity = (row['policy_id'], row['reference_id'], beta)
        if row['policy_id'] == row['reference_id']:
            raise ValueError('policy_and_reference_roles_must_differ')
        if identities is None:
            identities = identity
        if identity != identities:
            raise ValueError('inconsistent_run_identity')
        pairs = row['pairs']
        ids = [p['id'] for p in pairs]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('duplicate_or_empty_heldout_pairs')
        if expected_ids is None:
            expected_ids = set(ids)
        if set(ids) != expected_ids:
            raise ValueError('heldout_coverage_mismatch')
        margins = []
        for p in pairs:
            values = [p[k] for k in ('policy_chosen', 'reference_chosen',
                                    'policy_rejected', 'reference_rejected')]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v > 0
                   for v in values):
                raise ValueError('invalid_completion_log_probability')
            pc, rc, pr, rr = values
            margins.append(beta * ((pc - rc) - (pr - rr)))
        win = sum(m > 0 for m in margins) / len(margins)
        loss = sum(max(-m, 0) + math.log1p(math.exp(-abs(m)))
                   for m in margins) / len(margins)
        if (not math.isclose(win, row['rewards/accuracies'])
                or not math.isclose(sum(margins) / len(margins), row['rewards/margins'])
                or not math.isclose(loss, row['eval_loss'], abs_tol=1e-8)):
            raise ValueError('recorded_aggregate_mismatch')
        summaries.append({'step': row['step'], 'win_rate': win,
                          'mean_margin': sum(margins) / len(margins),
                          'loss': loss, 'ties': sum(m == 0 for m in margins),
                          'n': len(margins)})
    outcomes = {}
    ticket_ids = None
    for stage in ('before', 'after'):
        group = [r for r in replies if r['stage'] == stage]
        ids = {r['ticket_id'] for r in group}
        if not group or len(ids) != len(group):
            raise ValueError('missing_or_duplicate_ticket_review')
        if ticket_ids is None:
            ticket_ids = ids
        if ids != ticket_ids:
            raise ValueError('unpaired_ticket_evaluation')
        for r in group:
            if (r['source'] != RECORDED or r['split'] != 'held_out'
                    or r['review_source'] != 'independent_human_rubric'
                    or any(type(r[k]) is not bool for k in ('resolved', 'appropriate', 'generic'))):
                raise ValueError('independent_review_required')
        outcomes[stage] = {k + '_rate': sum(r[k] for r in group) / len(group)
                           for k in ('resolved', 'appropriate', 'generic')}
        outcomes[stage]['n'] = len(group)
    updated = [r for r in summaries if r['step'] > 0]
    proxy_up = len(updated) > 1 and updated[-1]['win_rate'] > updated[0]['win_rate']
    divergence = proxy_up and outcomes['after']['appropriate_rate'] < outcomes['before']['appropriate_rate']
    return {'source': RECORDED, 'evidence_scope': 'offline_evidence_only',
            'preference': summaries, 'resolved_ticket_eval': outcomes,
            'reward_hacking_flag': divergence, 'deployment_approved': False}
