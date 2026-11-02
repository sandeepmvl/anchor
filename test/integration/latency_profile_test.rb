require 'test_helper'
require 'rack/mock'
require 'open3'

class LatencyProfileTest < ActiveSupport::TestCase
  class ProbeController < ChatController
    class_attribute :profile_mode, default: 'chat'

    def create
      if profile_mode == 'chat'
        AiTelemetry.measure('model') { bridge('chat') }
      else
        trusted = RetrievalClient::TrustedContext.new(tenant_id: 1, conversation_id: 2)
        client = RetrievalClient.new(context: trusted, transport: lambda do |_payload, headers|
          bridge('rag', headers)
          {'reply' => 'offline', 'citations' => [1], 'abstained' => false}
        end)
        headers = AiTelemetry.current.headers
        client.complete(params[:message], request_id: headers.fetch('X-Request-Id'),
          traceparent: headers.fetch('traceparent'))
      end
      render json: {reply: 'offline'}
    end

    private

    def bridge(mode, headers = AiTelemetry.current.headers)
      output, error, status = Open3.capture3('python3', 'ops/load/profile_tickets.py',
        '--bridge-traceparent', headers.fetch('traceparent'), '--mode', mode,
        chdir: Rails.root.to_s)
      raise "offline Python probe failed: #{error}" unless status.success?
      report = JSON.parse(output)
      AiTelemetry.sink.call(event: 'python.profile_probe', report: report)
    end
  end

  test 'chat and RAG preserve one trace without ticket text or customer PII' do
    records = []
    previous_sink = AiTelemetry.sink
    AiTelemetry.sink = ->(record) { records << record }
    begin
      %w[chat rag].each do |mode|
        records.clear
        ProbeController.profile_mode = mode
        env = Rack::MockRequest.env_for('/chat', method: 'POST',
          input: 'message=private-ticket-alice%40example.com',
          'CONTENT_TYPE' => 'application/x-www-form-urlencoded')
        env['action_dispatch.request_id'] = 'latency-request'
        env['HTTP_TRACEPARENT'] = "00-#{'a' * 32}-#{'b' * 16}-01"
        status, _headers, body = ProbeController.action(:create).call(env)
        begin
          text = +''
          body.each { |chunk| text << chunk }
          assert_equal 200, status
          assert_equal 'offline', JSON.parse(text).fetch('reply')
        ensure
          body.close if body.respond_to?(:close)
        end
        spans = records.select { |r| r[:event] == 'span' }
        assert spans.any? { |r| r[:name] == 'rails.action' }
        assert spans.all? { |r| r[:trace_id] == 'a' * 32 }
        assert spans.all? { |r| r[:exclusive_ms] >= 0 && r[:inclusive_ms] >= r[:exclusive_ms] }
        python = records.find { |r| r[:event] == 'python.profile_probe' }.fetch(:report)
        assert_equal 'a' * 32, python.fetch('trace_id')
        assert python.fetch('spans').all? { |r| r.fetch('trace_id') == 'a' * 32 }
        encoded = JSON.generate(records)
        refute_includes encoded, 'private-ticket'
        refute_includes encoded, 'alice@example.com'
        refute_includes encoded, 'message='
      end
    ensure
      AiTelemetry.sink = previous_sink
    end
  end
end
