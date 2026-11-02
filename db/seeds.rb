# Required application data; safe to execute repeatedly.
# Existing published contracts must match rather than being overwritten.
{
  'chat:v1' => PromptRegistry::V1,
  'chat:v2' => PromptRegistry::V2
}.each do |version, attributes|
  PromptRegistry.publish(version, attributes)
end

if ENV['ANCHOR_HARDENED_PROMPT_CANDIDATE']
  PromptRegistry.harden_active(version: ENV.fetch('ANCHOR_HARDENED_PROMPT_CANDIDATE'))
end

unless Rails.env.production?
  baseline = Rails.root.join('eval/datasets/golden_v1.jsonl')
  File.foreach(baseline) do |line|
    row = JSON.parse(line)
    attributes = {tenant_id: row.fetch('tenant_id'), kind: 'example', key: row.fetch('source_run_id')}
    payload = row.except('tenant_id')
    existing = GoldenDataset.find_by(attributes)
    if existing
      # A later reviewed revision is allowed; the historical v1 stays frozen.
      raise 'golden_seed_identity_conflict' unless existing.payload['provenance'] == payload['provenance']
    else
      GoldenDataset.create!(**attributes, payload: payload)
    end
  end
  unless GoldenDataset.versions.exists?(tenant_id: 'tenant-demo', key: 'v1')
    docs = GoldenDatasets::Versioner.replay_evidence(baseline, tenant: 'tenant-demo')
    GoldenDatasets::Versioner.new(tenant: 'tenant-demo').publish!(name: 'v1', against: nil,
      docs: docs, built_against_tag: 'golden-baseline-v1')
  end
  puts 'Golden baseline registered as immutable v1 (replay tenant only).'

  [['Authorization Acme', 'A104'], ['Authorization Beta', 'B204']].each do |name, reference|
    organization = Organization.find_or_create_by!(name: name)
    organization.orders.find_or_create_by!(reference: reference) do |order|
      order.payment_id = "offline_#{reference}"
      order.refund_limit_cents = 500
    end
  end
end
