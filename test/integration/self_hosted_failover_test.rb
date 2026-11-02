require "minitest/autorun"
require "json"
require "open3"

class SelfHostedFailoverTest < Minitest::Test
  def test_offline_chat_boundary_preserves_policy_and_committed_actions
    output, error, status = Open3.capture3(
      "python3", "-m", "ops.deploy.self_hosted_release", "demo"
    )
    assert status.success?, error
    result = JSON.parse(output)
    assert_equal "local", result.fetch("healthy").fetch("provider")
    assert_equal "primary", result.fetch("outage").fetch("provider")
    %w[local_only human_only].each do |key|
      handoff = result.fetch(key)
      assert handoff.fetch("human_handoff")
      assert_equal [], handoff.fetch("tool_calls")
      refute_includes JSON.generate(handoff), "me@example.com"
      assert_equal ["refund-R42"], handoff.fetch("context").fetch("committed_actions")
    end
    assert_equal 1, result.fetch("fixture_charge_count")
    assert_equal ["refund-R42"], result.fetch("committed_actions")
    assert_equal "previous", result.fetch("rollback").fetch("profile")
  end
end
