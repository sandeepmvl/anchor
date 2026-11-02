# Run directly: ruby test/integration/production_launch_test.rb [--deployed]
require 'rbconfig'

if ARGV.delete('--deployed')
  root = File.expand_path('../..', __dir__)
  suites = %w[
    test/integration/security_tenant_isolation_test.rb
    test/integration/tool_authorization_test.rb
    test/integration/ai_quota_enforcement_test.rb
    test/integration/pii_pipeline_test.rb
    test/integration/safe_tool_actions_test.rb
    test/integration/handoff_drop_test.rb
    test/integration/realtime_voice_flow_test.rb
    test/services/ai_telemetry_test.rb
    test/integration/release_pointer_rollback_test.rb
    test/integration/state_recovery_test.rb
    test/integration/human_only_mode_test.rb
    test/agents/bounded_agent_test.rb
  ]
  Dir.chdir(root) do
    exit(system(RbConfig.ruby, 'bin/rails', 'test', *suites) ? 0 : 1)
  end
end

require 'minitest/autorun'
require 'open3'

class ProductionLaunchContractTest < Minitest::Test
  def test_interview_fix_preserves_terminal_escalation
    root = File.expand_path('../..', __dir__)
    output, status = Open3.capture2e(RbConfig.ruby,
      'test/agents/bounded_agent_test.rb', chdir: root)
    assert status.success?, output
  end

  def test_real_infrastructure_is_required
    root = File.expand_path('../..', __dir__)
    output, status = Open3.capture2e(ENV.fetch('ANCHOR_PYTHON', 'python3'), '-c',
      'from ops.deploy.validate_manifest import release_gate; release_gate()', chdir: root)
    refute status.success?
    assert_includes output, 'container and live-provider evidence missing'
  end

  def test_receipt_never_claims_deployed_fixture_traffic
    root = File.expand_path('../..', __dir__)
    source = File.read(File.join(root, 'bin/deploy'))
    assert_includes source, "scope='offline-rehearsal'"
    assert_includes source, 'deployment_approved=False'
    assert_includes source, 'live authentication and tenant traffic'
  end
end
