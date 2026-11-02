require 'minitest/autorun'
require 'open3'

class BoundedAgentContractTest < Minitest::Test
  def test_exhaustion_requires_scoped_terminal_human_handoff
    root = File.expand_path('../..', __dir__)
    output, status = Open3.capture2e(
      ENV.fetch('ANCHOR_PYTHON', 'python3'), '-m', 'pytest', '-q',
      'anchor_ai/tests/test_bounded_runtime.py',
      '-k', 'staff_terminal_escalation_contract', chdir: root
    )
    assert status.success?, output
    assert_includes output, '2 passed'
    refute_includes output, 'failed'
  end
end
