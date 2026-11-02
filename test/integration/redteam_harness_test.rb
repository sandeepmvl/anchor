require 'minitest/autorun'
require 'open3'
require 'json'

class RedteamHarnessTest < Minitest::Test
  ROOT = File.expand_path('../..', __dir__)

  def launch(*args, environment: 'test')
    Open3.capture3({ 'RAILS_ENV' => environment,
                    'ANCHOR_REDTEAM_ISOLATED' => '1',
                    'ANCHOR_REDTEAM_ADAPTER' => nil },
                  'ruby', File.join(ROOT, 'bin/redteam_harness.rb'), *args,
                  chdir: ROOT)
  end

  def report(output)
    JSON.parse(output[output.index('{')..])
  end

  def test_baseline_finding_disappears_without_dropping_trials
    baseline_out, _, baseline_status = launch('--baseline')
    hardened_out, _, hardened_status = launch
    refute baseline_status.success?
    assert hardened_status.success?
    baseline = report(baseline_out)
    hardened = report(hardened_out)
    assert_equal baseline['trials'].map { |r| r['id'] }, hardened['trials'].map { |r| r['id'] }
    %w[bounded_agent rag_generation].each do |route|
      finding = baseline['trials'].find { |r| r['id'] == "#{route}:indirect:original" }
      assert_includes finding['issues'], 'unauthorized_proposals'
      assert_equal 'high', finding['severity']
    end
    assert hardened['passed']
    refute hardened['release_verified']
    puts 'A-104: reproducible security verdict backed by retrieval and tool-state evidence.'
    puts 'Baseline finding detected; hardened contract fixtures block it; live defenses NOT verified.'
  end

  def test_baseline_cannot_run_in_production
    _, error, status = launch('--baseline', environment: 'production')
    refute status.success?
    assert_includes error, 'baseline_requires_isolated_test_process'
  end

  def test_unauthorized_effect_fails_gate
    output, _, status = launch('--fault')
    refute status.success?
    assert_includes report(output)['trials'].first['issues'], 'unauthorized_effects'
  end
end
