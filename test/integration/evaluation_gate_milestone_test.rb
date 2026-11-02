require "minitest/autorun"
require "open3"

class EvaluationGateMilestoneTest < Minitest::Test
  ROOT = File.expand_path("../..", __dir__)

  def invoke(*arguments)
    Open3.capture3(ENV.fetch("ANCHOR_PYTHON", "python3"), "-m", "eval.release_gate",
                  "smoke", *arguments, chdir: ROOT)
  end

  def test_approved_refund_passes_complete_offline_verdict
    output, error, status = invoke
    assert status.success?, error
    assert_includes output, "Anchor evaluation: PASS"
    assert_includes output, "OFFLINE ONLY"
  end

  def test_regressed_refund_is_red_without_changing_existing_acceptance_criteria
    output, _error, status = invoke("--regressed")
    refute status.success?
    assert_includes output, "| trajectory | failed | refund-unconfirmed | 0 |"
    assert_includes output, "| regression | failed | refund-unconfirmed | 0 |"
  end
end
