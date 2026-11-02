require_relative '../../labs/golden-dataset-construction-and-maintenance/test_helper'

class GoldenDatasetTest < GoldenTestCase
  def test_required_contract_and_tenant_scope
    valid = JSON.parse(File.foreach(replay).first).except('tenant_id')
    %w[query context expected_output tags difficulty source review_status provenance doc_version].each do |field|
      row = GoldenDataset.new(tenant_id: TENANT, kind: 'example', key: 'probe', payload: valid.except(field))
      refute row.valid?, "accepted missing #{field}"
    end
    row = GoldenDataset.new(tenant_id: nil, kind: 'example', key: 'probe', payload: valid)
    refute row.valid?
    row.tenant_id = 'another-tenant'
    refute row.valid?
    row.tenant_id = TENANT
    row.payload = valid.merge('query' => 'Email alex@example.test')
    refute row.valid?
    row.payload = valid.merge('status' => 'active', 'review_status' => 'pending')
    refute row.valid?
  end

  def test_versions_are_immutable_even_through_sql
    version = baseline
    assert_raises(ActiveRecord::ReadOnlyRecord) { version.update!(payload: version.payload.merge('built_at' => 'changed')) }
    assert_raises(ActiveRecord::ReadOnlyRecord) { version.destroy! }
    assert_raises(ActiveRecord::StatementInvalid) { version.update_columns(payload: {}) }
    assert_raises(ActiveRecord::StatementInvalid) { GoldenDataset.where(id: version.id).delete_all }
    assert_equal 'v1', version.reload.key
  end

  def test_identity_cannot_be_moved_to_another_tenant
    @builder.call(replay)
    row = GoldenDataset.examples.first
    assert_raises(ActiveRecord::StatementInvalid) { row.update_columns(tenant_id: 'other') }
  end

  def test_only_rag_examples_are_supported_by_this_increment
    payload = JSON.parse(File.foreach(replay).first).except('tenant_id').merge('case_type' => 'agent')
    row = GoldenDataset.new(tenant_id: TENANT, kind: 'example', key: 'agent-probe', payload: payload)
    refute row.valid?
  end

  def test_sanitizer_preserves_audit_metadata_and_is_idempotent
    metadata = {'intake_sha256' => '1234567890123456abcdef',
      'sha256' => 'abcdef1234567890123456', 'labeled_at' => '2026-01-01T00:00:00Z',
      'built_at' => '2026-01-01T00:00:00Z', 'deprecated_at' => '2026-01-01T00:00:00Z'}
    input = metadata.merge('query' => 'Email alex@example.test; phone +1 415 555 0123; card 4111 1111 1111 1111')
    safe = GoldenDataset.sanitize(input)
    assert_equal metadata, safe.slice(*metadata.keys)
    assert_equal 'Email [EMAIL]; phone [PHONE]; card [CARD]', safe['query']
    assert_equal safe, GoldenDataset.sanitize(safe)
  end
end
