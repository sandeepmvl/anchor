require 'minitest/autorun'
require 'open3'

# Operations integration only: the Python fixture is an explicit projection,
# not a substitute for Rails model/request integration coverage.
class StateRecoveryTest < Minitest::Test
  def test_isolated_restore_preserves_the_recovery_contract
    root = File.expand_path('../..', __dir__)
    output, status = Open3.capture2e(
      'python3', '-m', 'unittest',
      'anchor_ai.tests.test_backup_integrity.BackupIntegrityTest.test_isolated_restore_contract',
      chdir: root
    )
    assert status.success?, output
    assert_includes output, 'OK'
  end
end
