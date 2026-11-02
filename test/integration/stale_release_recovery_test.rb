require 'minitest/autorun'
require 'open3'
require 'json'

class StaleReleaseRecoveryTest < Minitest::Test
  ROOT = File.expand_path('../..', __dir__)

  def test_stale_registry_recovery_and_provider_safety
    output, error, status = Open3.capture3('python3',
      'ops/drills/stale_release_pointer.py', 'smoke', chdir: ROOT)
    assert status.success?, output + error
    assert_includes output, 'green liveness; failed semantic probe'
    assert_includes output, 'provider safety circuit stayed open'
    result = JSON.parse(File.read(File.join(ROOT, 'tmp/stale-release/golden-result.json')))
    assert result.fetch('passed')
    refute result.fetch('deployment_approved')
  end
end
