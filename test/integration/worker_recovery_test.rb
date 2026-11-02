require 'minitest/autorun'
require 'tmpdir'
require_relative '../../labs/production-operations/worker_recovery'

class WorkerRecoveryTest < Minitest::Test
  def setup
    @directory = Dir.mktmpdir('anchor-worker-recovery-')
    @drill = AnchorOperations::Drill.new(@directory)
  end

  def teardown
    FileUtils.remove_entry(@directory)
  end

  def test_stall_alert_graceful_drain_and_no_repeated_fixture_actions
    alert = @drill.setup
    assert_equal 'QueueAgeHigh', alert[:alert]
    assert_equal 2, alert[:samples].length
    assert_equal [0, 0], alert[:samples].map { |s| s[:completed] }
    assert_equal 20, alert[:samples].last[:pending]
    assert_operator alert[:samples].last[:oldest_pending_ms], :>=, 2000
    assert_nil alert[:completed_p95_ms]
    recovered = @drill.restart
    assert_equal 0, recovered[:pending]
    assert_equal 20, recovered[:completed]
    before = @drill.counts
    @drill.complete('backlog-0') # Retry with the original durable action key.
    assert_equal before, @drill.counts
    assert_equal({ tickets: 20, actions: 20 }, before)
    report = @drill.verify
    assert report[:passed]
    assert_equal 20, report[:fresh_samples]
    assert_operator report[:backlog_p95_ms], :>=, 2000
    assert_operator report[:fresh_p95_ms], :<, 2000
    assert_operator report[:elapsed_seconds], :<, 300
    error = assert_raises(RuntimeError) { @drill.production_gate! }
    assert_match(/fixture evidence is insufficient/, error.message)
  end

  def test_production_gate_refuses_missing_infrastructure_evidence
    error = assert_raises(RuntimeError) { @drill.production_gate! }
    assert_match(/Production recovery refused/, error.message)
    assert_match(/claim reconciliation/, error.message)
  end

  def test_maintenance_policy_has_owners_without_claiming_execution
    result = AnchorOperations.maintenance
    assert result[:policy_valid]
    refute result[:executed]
    assert_equal 'security_owner', result[:tasks].fetch('credential_rotation').fetch('owner')
  end
end
