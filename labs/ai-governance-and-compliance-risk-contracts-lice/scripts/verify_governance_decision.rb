require 'json'

def verify(condition, message)
  raise message unless condition
end

GovernanceDecision.transaction do
  record = GovernanceDecision.create!(
    tenant_ref: 'retail-eu',
    actor_ref: 'reviewer-17',
    decision: 'governance_review',
    status: 'held_pending_evidence',
    basis_refs: ['RC-7', 'DPA-2', 'DPA-5', 'EM-1', 'ED-1'],
    artifact_version: 'governance-bundle-v1'
  )
  record.reload
  verify(GovernanceDecision.for_tenant('retail-eu').exists?(record.id), 'own tenant record missing')
  verify(!GovernanceDecision.for_tenant('other-tenant').exists?(record.id), 'cross-tenant query leak')

  rejected = false
  begin
    GovernanceDecisionSerializer.call(record, tenant_ref: 'other-tenant')
  rescue SecurityError
    rejected = true
  end
  verify(rejected, 'serializer accepted wrong tenant')

  event = GovernanceDecisionSerializer.call(record, tenant_ref: 'retail-eu')
  expected = %i[event_type tenant_ref actor_ref decision status basis_refs artifact_version timestamp]
  verify(event.keys.sort == expected.sort, 'unexpected event fields')
  encoded = JSON.generate(event)
  verify(!encoded.include?('Where is my delivery?'), 'customer content in event')

  invalid = record.dup
  invalid.basis_refs = ['Where is my delivery?']
  verify(!invalid.valid? && invalid.errors[:basis_refs].any?, 'customer text accepted as reference')

  register = Rails.root.join('docs/governance/open_questions.md').read
  %w[RC-7 DPA-2 DPA-5 EM-1 ED-1].each do |reference|
    verify(register.include?(reference), "missing register issue #{reference}")
  end
  puts JSON.pretty_generate(event)
  puts 'PASS tenant boundary, minimized event, and register'
  raise ActiveRecord::Rollback
end
