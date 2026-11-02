require 'test_helper'
require 'json'
require 'net/http'
require 'open3'
require 'timeout'

class HandoffDropTest < ActionDispatch::IntegrationTest
  def with_adapter(mode)
    stdin, stdout, stderr, waiter = Open3.popen3(
      ENV.fetch('PYTHON', 'python3'), '-B',
      Rails.root.join('labs/handoff-context-debug/drop_context_fault.py').to_s,
      '--serve', '--mode', mode,
      chdir: Rails.root.to_s
    )
    stdin.close
    line = Timeout.timeout(5) { stdout.gets }
    assert line, 'Fault adapter did not publish its endpoint'
    config = JSON.parse(line)
    assert_equal(mode == 'drop', config.fetch('context_dropping_enabled'))
    uri = URI(config.fetch('endpoint'))
    assert_equal '127.0.0.1', uri.host
    yield uri
  ensure
    if waiter&.alive?
      Process.kill('TERM', waiter.pid)
      Timeout.timeout(3) { waiter.value }
    end
    stdout&.close
    stderr&.close
  end

  def request_adapter(uri, incomplete: false)
    client = Net::HTTP.new(uri.host, uri.port, nil)
    client.open_timeout = 2
    client.read_timeout = 3
    request = Net::HTTP::Post.new(uri.request_uri)
    request['Content-Type'] = 'application/json'
    request.body = JSON.generate(dry_run: true, confirmed: true, incomplete: incomplete)
    if defined?(WebMock)
      webmock_config = WebMock::Config.instance
      previous_allow = webmock_config.allow
      webmock_config.allow = lambda do |target|
        target.scheme == uri.scheme && target.host == uri.host && target.port == uri.port
      end
    end
    response = client.request(request)
    assert_equal '200', response.code
    JSON.parse(response.body)
  ensure
    if webmock_config
      webmock_config.allow = previous_allow
    end
  end

  def assert_original_order(result)
    triage = result.fetch('trace').find { |row| row['event'] == 'triage.exit' }
    entry = result.fetch('trace').find { |row| row['event'] == 'specialist.entry' }
    assert triage.fetch('order_id_present')
    assert entry.fetch('order_id_present'),
      'order_id at triage exit but absent at specialist entry'
    assert_equal 'A104', result.fetch('specialist_state').fetch('order_id')
    tool = result.fetch('tool_calls').fetch(0)
    assert_equal 'issue_refund', tool.fetch('name')
    assert_equal({'order_id' => 'A104', 'amount_cents' => 300}, tool.fetch('arguments'))
    assert_equal true, tool.fetch('dry_run')
    assert_equal 'offline_test_double', tool.fetch('backend')
  end

  test 'original order reaches specialist and guarded refund tool or exposes planted fault' do
    reproduce = ENV['ANCHOR_HANDOFF_REPRODUCE'] == '1'
    with_adapter(reproduce ? 'drop' : 'fixed') do |uri|
      result = request_adapter(uri)
      if reproduce
        error = assert_raises(Minitest::Assertion) { assert_original_order(result) }
        assert_includes error.message, 'order_id at triage exit but absent at specialist entry'
        assert_equal 0, result.fetch('tool_executions')
        puts "Expected diagnostic failure: #{error.message}"
      else
        assert_original_order(result)
      end
    end
  end

  test 'incomplete handoff fails closed before any tool executes' do
    with_adapter('fixed') do |uri|
      result = request_adapter(uri, incomplete: true)
      assert_equal 'human_escalation', result.fetch('status')
      assert_equal 0, result.fetch('tool_executions')
      assert_empty result.fetch('tool_calls')
      assert_equal 'A104', result.fetch('context').fetch('order_id')
      refute result.fetch('trace').any? { |row| row['event'] == 'specialist.admitted' }
    end
  end
end
