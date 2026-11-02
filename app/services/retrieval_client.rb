require 'json'
require 'net/http'
require 'uri'
require 'securerandom'

class RetrievalClient
  class Error < StandardError; end

  class TrustedContext
    attr_reader :tenant_id, :conversation_id

    def initialize(tenant_id:, conversation_id:)
      unless [tenant_id, conversation_id].all? { |id| id.is_a?(Integer) && id.positive? }
        raise Error, 'trusted_context_required'
      end
      @tenant_id, @conversation_id = tenant_id, conversation_id
      freeze
    end
  end

  def initialize(context:, transport: nil, url: nil, secret: nil)
    raise Error, 'trusted_context_required' unless context.is_a?(TrustedContext)
    @context = context
    @transport = transport
    @url = url
    @secret = secret
  end

  def complete(message, request_id:, traceparent:)
    unless request_id.is_a?(String) && /\A[A-Za-z0-9_-]{1,80}\z/.match?(request_id)
      raise Error, 'invalid_request_id'
    end
    unless /\A00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}\z/.match?(traceparent.to_s)
      raise Error, 'invalid_traceparent'
    end
    payload = { message: message.to_s, tenant_id: @context.tenant_id,
                conversation_id: @context.conversation_id }
    headers = { 'X-Request-Id' => request_id, 'traceparent' => traceparent }
    result = @transport ? @transport.call(payload, headers) : perform(payload, headers)
    valid = result.is_a?(Hash) && result['reply'].is_a?(String) &&
      result['citations'].is_a?(Array) &&
      result['citations'].all? { |id| id.is_a?(Integer) && id.positive? } &&
      [true, false].include?(result['abstained'])
    raise Error, 'invalid_retrieval_response' unless valid
    result
  end

  private

  def perform(payload, headers)
    uri = URI(@url.to_s)
    raise Error, 'invalid_retrieval_url' unless %w[http https].include?(uri.scheme) && uri.host
    raise Error, 'retrieval_secret_required' unless @secret.is_a?(String) && !@secret.empty?
    request = Net::HTTP::Post.new(uri.request_uri)
    request['Content-Type'] = 'application/json'
    request['X-Anchor-Secret'] = @secret
    headers.each { |key, value| request[key] = value }
    request.body = JSON.generate(payload)
    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = uri.scheme == 'https'
    http.open_timeout = 2
    http.read_timeout = 8
    http.max_retries = 0
    response = http.request(request)
    raise Error, 'retrieval_upstream_failed' unless response.code.to_i == 200
    JSON.parse(response.body)
  rescue JSON::ParserError, URI::InvalidURIError
    raise Error, 'invalid_retrieval_response'
  end
end
