require 'test_helper'
require 'webmock/minitest'

WebMock.disable_net_connect!

class AiTelemetryTest < ActiveSupport::TestCase
  setup do
    @keys = %w[ANCHOR_AI_SECRET ANCHOR_AI_URL ANCHOR_AI_OPEN_TIMEOUT ANCHOR_AI_READ_TIMEOUT]
    @previous = @keys.to_h { |key| [key, ENV[key]] }
    ENV['ANCHOR_AI_SECRET'] = 'SERVICE_SECRET_SENTINEL'
    ENV['ANCHOR_AI_URL'] = 'http://127.0.0.1:8000/v1/infer'
    ENV['ANCHOR_AI_OPEN_TIMEOUT'] = '2'
    ENV['ANCHOR_AI_READ_TIMEOUT'] = '5'
    @records = []
    @sink = AiTelemetry.sink
    AiTelemetry.sink = ->(record) { @records << JSON.parse(JSON.generate(record)) }
  end

  teardown do
    AiTelemetry.sink = @sink
    @previous.each { |key, value| value.nil? ? ENV.delete(key) : ENV[key] = value }
  end

  test 'boundary forwards context and records duration and outcome' do
    context = AiTelemetry.context(request_id: 'ticket-0',
      traceparent: '00-1234567890abcdef1234567890abcdef-aaaaaaaaaaaaaaaa-01')
    sent = nil
    stub_request(:post, ENV.fetch('ANCHOR_AI_URL')).to_return do |request|
      sent = request.headers
      { status: 200, body: '{"reply":"safe reply","tokens":38}' }
    end
    AiTelemetry.span('http.chat', context: context) do
      assert_equal 'safe reply', LlmClient.new(provider: 'http').complete('PRIVATE_MESSAGE_SENTINEL')
    end
    boundary, root = @records
    assert_equal 'rails.infer_call', boundary['name']
    assert_equal root['span_id'], boundary['parent_span_id']
    assert_equal context[:trace_id], boundary['trace_id']
    assert_equal 'ticket-0', sent['X-Request-Id']
    assert_equal "00-#{boundary['trace_id']}-#{boundary['span_id']}-01", sent['Traceparent']
    assert_equal 'ok', boundary['status']
    assert_operator boundary['latency_ms'], :>=, 0
    text = JSON.generate(@records)
    %w[PRIVATE_MESSAGE_SENTINEL SERVICE_SECRET_SENTINEL Authorization].each do |sentinel|
      refute_includes text, sentinel
    end
    puts JSON.generate(boundary)
  end

  test 'timeout closes the span and restores context' do
    stub_request(:post, ENV.fetch('ANCHOR_AI_URL')).to_raise(Net::ReadTimeout)
    assert_raises(LlmClient::BoundaryError) do
      LlmClient.new(provider: 'http').complete('PRIVATE_MESSAGE_SENTINEL')
    end
    item = @records.fetch(0)
    assert_equal 'error', item['status']
    assert_equal 'ai_read_timeout', item['error_type']
    assert_operator item['latency_ms'], :>=, 0
    assert_nil AiTelemetry.current
    refute_includes JSON.generate(@records), 'PRIVATE_MESSAGE_SENTINEL'
  end
end
