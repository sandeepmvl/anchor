require 'json'
require 'securerandom'
require 'time'

class AiTelemetry
  FIELDS = %i[stage status_code tenant_id conversation_id].freeze
  ERRORS = %w[ai_auth_rejected ai_contract_rejected ai_upstream_error ai_invalid_response
    ai_connect_timeout ai_read_timeout ai_connection_refused ai_configuration_error
    provider_bad_request provider_auth_failed provider_forbidden provider_not_found
    provider_quota_exhausted provider_rate_limited provider_unknown_429 provider_unavailable
    provider_transport_error provider_timeout invalid_provider_response missing_provider_key
    invalid_provider_mode provider_rejected internal_error
    trusted_context_required retrieval_upstream_failed invalid_retrieval_response
    invalid_retrieval_url retrieval_secret_required invalid_request_id
    invalid_traceparent retrieval_stream_not_supported].freeze

  class << self
    attr_writer :sink

    def sink
      @sink ||= ->(record) { $stdout.puts(JSON.generate(record)) }
    end

    def golden_event(status:, count:)
      unless %w[ok rejected].include?(status) && count.is_a?(Integer) && count >= 0
        raise ArgumentError, 'invalid_golden_diagnostic'
      end
      sink.call(event: 'golden_import', status: status, count: count)
    end

    def context(request_id: nil, traceparent: nil)
      match = /\A00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})\z/.match(traceparent.to_s)
      valid = match && match[1].to_i(16).positive? && match[2].to_i(16).positive?
      safe = request_id.is_a?(String) && /\A[A-Za-z0-9_-]{1,80}\z/.match?(request_id)
      { request_id: safe ? request_id : SecureRandom.uuid,
        trace_id: valid ? match[1] : SecureRandom.hex(16),
        parent_span_id: valid ? match[2] : nil }
    end

    def current
      Thread.current[:anchor_ai_span]
    end

    def span(name, context: nil, **fields)
      previous = current
      ctx = context || (previous ? previous.child_context : self.context)
      item = Span.new(name, ctx, **fields)
      Thread.current[:anchor_ai_span] = item
      begin
        yield item
      rescue StandardError => error
        item.fail(error.respond_to?(:code) ? error.code : 'internal_error',
          error.respond_to?(:status) ? error.status : nil)
        raise
      ensure
        Thread.current[:anchor_ai_span] = previous
        item.finish
      end
    end
  end

  class Span
    attr_reader :record

    def initialize(name, context, **fields)
      @record = { event: 'span', name: name, **context, span_id: SecureRandom.hex(8),
        started_at: Time.now.utc.iso8601(3), status: 'ok', error_type: nil,
        error_message: nil, tokens: nil, cost_estimate: nil }
      @started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      set(**fields)
    end

    def set(**fields)
      raise ArgumentError, 'non-allowlisted telemetry field' unless (fields.keys - FIELDS).empty?
      @record.merge!(fields)
    end

    def child_context
      @record.slice(:request_id, :trace_id).merge(parent_span_id: @record[:span_id])
    end

    def headers
      { 'X-Request-Id' => @record[:request_id],
        'traceparent' => "00-#{@record[:trace_id]}-#{@record[:span_id]}-01" }
    end

    def fail(code, status = nil)
      code = 'internal_error' unless ERRORS.include?(code)
      @record.merge!(status: 'error', error_type: code, error_message: code, status_code: status)
    end

    def finish
      @record.merge!(ended_at: Time.now.utc.iso8601(3),
        latency_ms: ((Process.clock_gettime(Process::CLOCK_MONOTONIC) - @started) * 1000).round(3))
      AiTelemetry.sink.call(@record.dup)
    end
  end
end
