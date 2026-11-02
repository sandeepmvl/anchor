require_relative '../../../labs/golden-dataset-construction-and-maintenance/test_helper'

class GoldenBuilderTest < GoldenTestCase
  def test_five_imports_are_idempotent_and_review_is_preserved
    assert_equal 5, @builder.call(replay)
    keys = GoldenDataset.examples.order(:key).pluck(:key)
    @builder.approve!(keys: keys, reviewer: 'reviewer-a')
    assert_equal 0, @builder.call(replay)
    assert_equal 5, GoldenDataset.examples.count
    assert GoldenDataset.examples.all? { |row| row.payload['labeler'] == 'reviewer-a' && row.payload['review_status'] == 'approved' }
    assert_equal 'replay', GoldenDataset.examples.first.payload.dig('provenance', 'kind')
  end

  def test_redacts_before_storage_and_diagnostics
    row = JSON.parse(File.foreach(replay).first)
    secrets = ['alex@example.test', '+1 415 555 0123', '4111 1111 1111 1111']
    row['query'] = secrets.join(' ')
    row['context'][0]['text'] += ' ' + secrets.join(' ')
    row['expected_output'] += ' ' + secrets.join(' ')
    with_rows([row]) { |path| assert_equal 1, @builder.call(path) }
    persisted = GoldenDataset.examples.first.payload.to_json
    diagnostics = @events.to_json
    secrets.each do |secret|
      refute_includes persisted, secret
      refute_includes diagnostics, secret
    end
    assert_includes persisted, '[EMAIL]'
    assert_includes persisted, '[PHONE]'
    assert_includes persisted, '[CARD]'
    assert_equal [{event: 'golden_import', status: 'ok', count: 1}], @events
  end

  def test_cross_tenant_context_rolls_back_the_entire_batch
    rows = File.readlines(replay).first(2).map { |line| JSON.parse(line) }
    rows[1]['context'][0]['tenant_id'] = 'private-tenant'
    with_rows(rows) { |path| assert_raises(ArgumentError) { @builder.call(path) } }
    assert_equal 0, GoldenDataset.examples.count
    assert_equal 'rejected', @events.last[:status]
    refute_includes @events.to_json, 'private-tenant'
    rows[0]['tenant_id'] = 'private-tenant'
    with_rows([rows[0]]) { |path| assert_raises(ArgumentError) { @builder.call(path) } }
  end

  def test_malformed_import_is_rejected_and_changed_source_is_not_overwritten
    bad = JSON.parse(File.foreach(replay).first).except('expected_output')
    with_rows([bad]) { |path| assert_raises(ArgumentError) { @builder.call(path) } }
    assert_equal 0, GoldenDataset.examples.count
    @builder.call(replay)
    changed = JSON.parse(File.foreach(replay).first).merge('expected_output' => 'Different label')
    with_rows([changed]) { |path| assert_raises(ArgumentError) { @builder.call(path) } }
    refute_equal 'Different label', GoldenDataset.examples.find_by!(key: 'release-run-1').payload['expected_output']
  end

  def test_telemetry_refuses_text_fields
    assert_raises(ArgumentError) { AiTelemetry.golden_event(status: 'ok', count: 'alex@example.test') }
    assert_raises(ArgumentError) { AiTelemetry.golden_event(status: 'ok', count: 1, query: 'private') }
  end
end
