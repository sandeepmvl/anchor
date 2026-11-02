require "json"
require "net/http"
require "uri"

class AgentOrchestrationClient
  class Unavailable < StandardError; end

  def initialize(url: ENV.fetch("ANCHOR_AI_URL", "http://127.0.0.1:8000"),
                 secret: ENV["ANCHOR_AI_SECRET"], transport: nil)
    @url = URI.join("#{url.delete_suffix('/')}/", "v1/multi_agent")
    raise ArgumentError, "invalid AI URL" unless %w[http https].include?(@url.scheme)
    @secret = secret
    @transport = transport || method(:send_request)
  end

  def execute(body, execution_context:, request_id:, traceparent:, idempotency_key: nil)
    raise Unavailable, "service_not_configured" if @secret.to_s.empty?
    raise ArgumentError, "missing_execution_context" if execution_context.to_s.empty?
    request = Net::HTTP::Post.new(@url)
    request["Content-Type"] = "application/json"
    request["X-Anchor-Secret"] = @secret
    request["X-Anchor-Context"] = execution_context
    request["X-Request-Id"] = request_id
    request["traceparent"] = traceparent
    request["Idempotency-Key"] = idempotency_key if idempotency_key
    request.body = JSON.generate(body)
    response = @transport.call(@url, request)
    raise Unavailable, "orchestration_unavailable" unless response.code.to_i.between?(200, 299)
    value = JSON.parse(response.body)
    raise Unavailable, "invalid_orchestration_response" unless value.is_a?(Hash)
    value
  rescue JSON::ParserError, IOError, SystemCallError, Timeout::Error
    raise Unavailable, "orchestration_transport_failed"
  end

  def answer_attachments(evidence, tenant_id:, conversation_id:)
    MultimodalClient.new.answer(evidence, tenant_id: tenant_id,
      conversation_id: conversation_id)
  end

  private

  def send_request(uri, request)
    Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https",
                   open_timeout: 2, read_timeout: 5) do |http|
      http.request(request)
    end
  end
end
