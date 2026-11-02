# The standalone exercise replaces ActiveRecord::Base's connection with a
# disposable registry database. Never do that inside the Rails test process.
if defined?(Rails)
  require 'test_helper'
  require 'open3'
  require 'rbconfig'

  class ReleasePointerRollbackIsolationTest < ActiveSupport::TestCase
    test 'release registry contracts pass in an isolated Ruby process' do
      output, status = Open3.capture2e(
        RbConfig.ruby, File.expand_path(__FILE__), chdir: Rails.root.to_s
      )
      assert status.success?, "Standalone release registry contracts failed:\n#{output}"
    end
  end
else
  require 'minitest/autorun'
  require_relative '../../labs/rollback-strategies/test_support'

class ReleasePointerRollbackTest < Minitest::Test
  def setup
    @directory = Dir.mktmpdir('anchor-release-test-')
    ReleaseLab.connect(File.join(@directory, 'registry.sqlite3'))
    @bundles = ReleaseLab.bundles
    AiReleaseRegistry.create!(tenant_id: 1, bundle: @bundles.fetch('good'), contract: 'support-tools:v1')
    @registry = ReleaseLab.registry
  end

  def teardown
    ActiveRecord::Base.connection_pool.disconnect!
    FileUtils.remove_entry(@directory)
  end

  def stage(name)
    @registry.stage(bundle: @bundles.fetch(name), percent: 20,
      expected_generation: @registry.pointer.generation, actor: 'test')
  end

  def test_failed_canary_cannot_advance_and_revert_restores_new_requests
    before = @registry.pin(conversation_id: 'same-ticket')
    row = stage('degraded')
    admitted = 100.times.map { |n| @registry.pin(conversation_id: n.to_s) }.find { |p| p['bundle_id'] == row.candidate }
    refute_nil admitted
    health = ReleaseLab.health(row).merge('candidate_quality' => 0.2)
    assert_raises(ArgumentError) { @registry.promote(expected_generation: row.generation, actor: 'test', health: health) }
    assert_equal before['bundle_id'], @registry.pointer.stable
    assert_equal row.generation, @registry.pointer.generation
    # Even optimistic online metrics cannot bypass the golden regression gate.
    error = assert_raises(ArgumentError) { @registry.promote(expected_generation: row.generation, actor: 'test', health: ReleaseLab.health(row)) }
    assert_equal 'golden_and_judge_gate_failed', error.message
    reverted = @registry.revert(expected_generation: row.generation, actor: 'test')
    assert_nil reverted.candidate
    100.times do |n|
      assert_equal before['bundle_id'], @registry.pin(conversation_id: n.to_s)['bundle_id']
    end
    assert_equal 'Order A17 has already been refunded.', admitted['bundle']['reply']
    assert_equal before['bundle']['reply'], @registry.pin(conversation_id: 'same-ticket')['bundle']['reply']
    assert_equal 'revert', reverted.audit.last.fetch('action')
    assert_equal 'test', reverted.audit.last.fetch('actor')
    assert_equal row.generation + 1, reverted.generation
  end

  def test_index_only_bundle_can_promote_and_revert_without_prompt_model_change
    original = @registry.pointer.stable
    row = stage('prior_index')
    row = @registry.promote(expected_generation: row.generation, actor: 'test', health: ReleaseLab.health(row))
    changed = @registry.pin(conversation_id: 'same-ticket')
    assert_equal 'support-index:v0', changed['bundle']['index']
    assert_equal @bundles['good']['prompt'], changed['bundle']['prompt']
    assert_equal @bundles['good']['model'], changed['bundle']['model']
    reverted = @registry.revert(expected_generation: row.generation, actor: 'test')
    assert_equal original, reverted.stable
    assert_equal 'support-index:v1', @registry.pin(conversation_id: 'same-ticket')['bundle']['index']
  end

  def test_incompatible_target_requires_roll_forward
    row = stage('prior_index')
    row = @registry.promote(expected_generation: row.generation, actor: 'test', health: ReleaseLab.health(row))
    # Model the separately authorized tool-schema migration.
    row.update!(required_contract: 'support-tools:v2')
    error = assert_raises(ArgumentError) { @registry.revert(expected_generation: row.generation, actor: 'test') }
    assert_equal 'incompatible_bundle_roll_forward_required', error.message
    assert_equal row.stable, @registry.pointer.stable
  end

  def test_stale_health_sparse_scores_and_production_refuse_release
    row = stage('prior_index')
    health = ReleaseLab.health(row).merge('candidate_count' => 0)
    assert_raises(ArgumentError) { @registry.promote(expected_generation: row.generation, actor: 'test', health: health) }
    assert_raises(ArgumentError) { @registry.revert(expected_generation: row.generation - 1, actor: 'test') }
    live = AiReleaseRegistry.new(tenant_id: 1)
    error = assert_raises(ArgumentError) { live.pin(conversation_id: 'same-ticket') }
    assert_equal 'production_readiness_evidence_required', error.message
    assert_raises(ActiveRecord::RecordNotFound) { ReleaseLab.registry(2).pin(conversation_id: 'same-ticket') }
  end
end
end
