require "net/http"
require "json"
require "yaml"
require "erb"

class LlmClient
  CANNED_REPLY = "Anchor stub: thanks for your message, a support reply is on its way."

  class BoundaryError < StandardError
    attr_reader :code, :status

    def initialize(code, status)
      @code, @status = code, status
      super(code)
    end
  end

  def initialize(provider: Rails.configuration.x.llm.fetch(:provider))
    @provider = provider
  end

  def complete(message)
    case @provider
    when "stub"
      CANNED_REPLY
    when "openai"
      api_key
      raise NotImplementedError, "Live calls arrive in the LLM provider APIs lesson"
    when "http"
      http_complete(message)
    else
      raise ArgumentError, "unknown provider #{@provider.inspect}"
    end
  end

  private

  def api_key
    key = ENV["OPENAI_API_KEY"]
    key = Rails.application.credentials.openai_api_key if key.nil? || key.empty?
    if key.nil? || key.empty?
      raise ArgumentError, "Set OPENAI_API_KEY before selecting the live provider"
    end
    key
  end

  def http_complete(message)
    config = YAML.safe_load(ERB.new(Rails.root.join("config/llm.yml").read).result).fetch("shared")
    uri = URI(config.fetch("url"))
    secret = ENV.fetch("ANCHOR_AI_SECRET")
    raise ArgumentError, "empty service secret" if secret.empty?
    unless %w[http https].include?(uri.scheme) && uri.host
      raise ArgumentError, "invalid service URL"
    end

    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = uri.scheme == "https"
    http.open_timeout = positive_timeout(config.fetch("open_timeout"))
    http.read_timeout = positive_timeout(config.fetch("read_timeout"))
    http.max_retries = 0
    request = Net::HTTP::Post.new(uri.request_uri)
    request["Content-Type"] = "application/json"
    request["X-Anchor-Secret"] = secret
    request.body = JSON.generate(
      version: "v1", conversation_id: 1, message: message, tenant_id: 1
    )
    response = http.request(request)
    status = response.code.to_i
    fail_boundary("ai_auth_rejected", 502) if status == 401
    fail_boundary("ai_contract_rejected", 502) if status == 422
    fail_boundary("ai_upstream_error", 502) unless (200..299).cover?(status)
    data = JSON.parse(response.body)
    valid = data.is_a?(Hash) && data["reply"].is_a?(String) &&
      data["tokens"].is_a?(Integer) && data["tokens"] >= 0
    fail_boundary("ai_invalid_response", 502) unless valid
    data.fetch("reply")
  rescue Net::OpenTimeout
    fail_boundary("ai_connect_timeout", 504)
  rescue Net::ReadTimeout
    fail_boundary("ai_read_timeout", 504)
  rescue Errno::ECONNREFUSED
    fail_boundary("ai_connection_refused", 503)
  rescue JSON::ParserError, TypeError
    fail_boundary("ai_invalid_response", 502)
  rescue KeyError, ArgumentError
    fail_boundary("ai_configuration_error", 503)
  end

  def positive_timeout(raw)
    value = Float(raw)
    raise ArgumentError, "invalid timeout" unless value.finite? && value.positive?
    value
  end

  def fail_boundary(code, status)
    raise BoundaryError.new(code, status)
  end
end

ActiveSupport.on_load(:action_controller_base) do
  rescue_from LlmClient::BoundaryError do |error|
    render json: { error: error.code }, status: error.status
  end
end
