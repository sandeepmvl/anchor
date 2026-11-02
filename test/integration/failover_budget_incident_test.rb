require 'minitest/autorun'
require 'tmpdir'
require 'fileutils'
require_relative '../../app/services/ai_quota_policy'

class FailoverBudgetIncidentTest < Minitest::Test
  def setup
    @directory = Dir.mktmpdir
    @path = File.join(@directory, 'incident.sqlite3')
    @context = { 'message' => 'Where is A17?', 'history' => ['customer question'],
      'evidence' => ['order:A17'], 'proposed_action' => 'lookup_order' }
    authority
  end

  def teardown
    FileUtils.remove_entry(@directory)
  end

  def authority(now = 1893456000)
    AiQuotaPolicy::IncidentWindow.new(path: @path, clock: -> { now })
  end

  def reserve(service, index)
    service.call('reserve', tenant: 1, ticket: "A17-#{index}", context: @context,
      attempt_id: "attempt-#{index}", bound: 1)
  end

  def test_independent_concurrent_requests_and_restart_share_window
    workers = 4.times.map do |worker|
      output = File.join(@directory, "worker-#{worker}.json")
      pid = fork do
        begin
          service = authority
          results = 4.times.map { |offset| reserve(service, worker * 4 + offset) }
          File.write(output, JSON.generate(results))
          exit! 0
        rescue StandardError => e
          warn e.full_message
          exit! 1
        end
      end
      [pid, output]
    end
    results = workers.flat_map do |pid, output|
      _, status = Process.wait2(pid)
      assert status.success?, 'concurrent authority worker failed'
      JSON.parse(File.read(output))
    end
    assert_equal 12, results.count { |r| r['allowed'] }
    restarted = authority
    refute reserve(restarted, 17)['allowed']
    snapshot = restarted.call('snapshot', tenant: 1)
    assert_equal 12, snapshot['used']
    assert_equal 12, snapshot['attempts']
    blocked = snapshot['tickets'].select { |t| t['state'] == 'human' }
    assert_equal 5, blocked.length
    assert blocked.all? { |t| t['context'] == @context }
    assert blocked.all? { |t| t['reason'] == 'budget_exhausted' }
    renewed = reserve(authority(1893456060), 18)
    assert renewed['allowed']
    assert_equal 0, renewed['used_before']
    assert_equal 13, renewed['attempts']
  end

  def test_human_only_hands_off_queued_and_new_tickets_without_reservations
    service = authority
    service.call('queue', tenant: 1, ticket: 'queued', context: @context)
    service.call('human_only', tenant: 1)
    result = reserve(authority, 99)
    refute result['allowed']
    assert_equal 'human_only', result['reason']
    snapshot = service.call('snapshot', tenant: 1)
    assert_equal 0, snapshot['used']
    assert_equal 0, snapshot['attempts']
    assert_equal 2, snapshot['tickets'].length
    assert snapshot['tickets'].all? { |t| t['state'] == 'human' && t['context'] == @context }
    other = service.call('snapshot', tenant: 2)
    refute other['human_only']
  end
end
