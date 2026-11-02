require "net/http"
require "json"
require "yaml"
require "erb"

class LlmClient
  CANNED_REPLY = "Anchor stub: thanks for your message, a support reply is on its way."

  class BoundaryError < StandardError
    attr_reader :code, :status, :retryable, :retry_after, :details

    def initialize(code, status, retryable: false, retry_after: nil, details: {})
      @code, @status = code, status
      @retryable, @retry_after = retryable, retry_after
      @details = details
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
      http_complete(message)
    when "http"
      http_complete(message)
    else
      raise ArgumentError, "unknown provider #{@provider.inspect}"
    end
  end

  def structured_reply(message)
    AiTelemetry.span('rails.infer_call', stage: 'boundary', tenant_id: 1, conversation_id: 1) do |span|
      perform_http_complete(message, span, structured: true)
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
    AiTelemetry.span('rails.infer_call', stage: 'boundary', tenant_id: 1, conversation_id: 1) do |span|
      perform_http_complete(message, span)
    end
  end

  def perform_http_complete(message, span, structured: false)
    config = YAML.safe_load(ERB.new(Rails.root.join("config/llm.yml").read).result).fetch("shared")
    uri = URI(config.fetch("url"))
    secret = ENV.fetch("ANCHOR_AI_SECRET")
    raise ArgumentError, "empty service secret" if secret.empty?
    unless %w[http https].include?(uri.scheme) && uri.host
      raise ArgumentError, "invalid service URL"
    end
    uri = URI.join(uri.to_s, "structured_replies") if structured

    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = uri.scheme == "https"
    http.open_timeout = positive_timeout(config.fetch("open_timeout"))
    http.read_timeout = positive_timeout(config.fetch("read_timeout"))
    http.max_retries = 0
    request = Net::HTTP::Post.new(uri.request_uri)
    request["Content-Type"] = "application/json"
    request["X-Anchor-Secret"] = secret
    span.headers.each { |key, value| request[key] = value }
    request.body = JSON.generate(
      version: "v1", conversation_id: 1, message: message, tenant_id: 1
    )
    response = http.request(request)
    status = response.code.to_i
    span.set(status_code: status)
    structured_failure(response) if structured && status == 422
    provider_failure(response, status) unless (200..299).cover?(status)
    fail_boundary("ai_auth_rejected", 502) if status == 401
    fail_boundary("ai_contract_rejected", 502) if status == 422
    fail_boundary("ai_upstream_error", 502) unless (200..299).cover?(status)
    data = JSON.parse(response.body)
    if structured
      valid = data.is_a?(Hash) && data.keys.sort == %w[category urgency] &&
        %w[billing technical account order].include?(data["category"]) &&
        %w[P1 P2 P3].include?(data["urgency"])
      fail_boundary("ai_invalid_response", 502) unless valid
      return data
    end
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

  def structured_failure(response)
    data = JSON.parse(response.body)
    detail = data.is_a?(Hash) ? data["detail"] : nil
    return unless detail.is_a?(Hash) && detail["code"] == "invalid_model_output"
    reason = detail["reason"]
    return unless %w[invalid_json schema_violation incomplete_output provider_refusal].include?(reason)
    details = { reason: reason }
    if reason == "schema_violation"
      errors = detail["errors"]
      return unless errors.is_a?(Array) && errors.length <= 10
      allowed_types = %w[missing literal_error extra_forbidden model_type string_type]
      return unless errors.all? { |e| e.is_a?(Hash) &&
        e["loc"].is_a?(Array) && e["loc"].length == 1 &&
        %w[category urgency $].include?(e["loc"][0]) && allowed_types.include?(e["type"]) }
      details[:errors] = errors.map { |e| { loc: e["loc"], type: e["type"] } }
    end
    raise BoundaryError.new("invalid_model_output", 422, details: details)
  end

  def provider_failure(response, status)
    data = JSON.parse(response.body)
    detail = data.is_a?(Hash) ? data["detail"] : nil
    return unless detail.is_a?(Hash)
    codes = %w[provider_bad_request provider_auth_failed provider_forbidden
      provider_not_found provider_quota_exhausted provider_rate_limited
      provider_unknown_429 provider_unavailable provider_transport_error
      provider_timeout invalid_provider_response missing_provider_key
      invalid_provider_mode provider_rejected]
    code = detail["code"]
    return unless codes.include?(code)
    retryable = detail["retryable"] == true && %w[provider_rate_limited
      provider_unavailable provider_transport_error provider_timeout].include?(code)
    raw = detail["retry_after"]
    wait = raw if retryable && raw.is_a?(Integer) && (0..60).cover?(raw)
    mapped = code == "provider_rate_limited" ? 429 : 502
    raise BoundaryError.new(code, mapped, retryable: retryable, retry_after: wait)
  rescue JSON::ParserError
    nil
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
    response.set_header("Retry-After", error.retry_after.to_s) unless error.retry_after.nil?
    body = { error: error.code }
    body.merge!(error.details) if error.code == "invalid_model_output"
    if error.code.start_with?("provider_")
      body.merge!(retryable: error.retryable, retry_after: error.retry_after)
    end
    render json: body, status: error.status
  end
end
