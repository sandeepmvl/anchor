require "test_helper"
require "webmock/minitest"
require "securerandom"

WebMock.disable_net_connect!

class AiBoundaryTest < ActiveSupport::TestCase
  setup do
    @keys = %w[ANCHOR_AI_SECRET ANCHOR_AI_URL ANCHOR_AI_OPEN_TIMEOUT ANCHOR_AI_READ_TIMEOUT]
    @previous = @keys.to_h { |key| [key, ENV[key]] }
    ENV["ANCHOR_AI_SECRET"] = SecureRandom.hex(24)
    ENV["ANCHOR_AI_URL"] = "http://127.0.0.1:8000/v1/infer"
    ENV["ANCHOR_AI_OPEN_TIMEOUT"] = "2"
    ENV["ANCHOR_AI_READ_TIMEOUT"] = "5"
    @client = LlmClient.new(provider: "http")
  end

  teardown do
    @previous.each { |key, value| value.nil? ? ENV.delete(key) : ENV[key] = value }
  end

  def upstream
    stub_request(:post, ENV.fetch("ANCHOR_AI_URL")).with(
      headers: {
        "X-Anchor-Secret" => ENV.fetch("ANCHOR_AI_SECRET"),
        "Content-Type" => "application/json"
      },
      body: JSON.generate(version: "v1", conversation_id: 1, message: "Hello", tenant_id: 1)
    ) do |request|
      /\A00-[0-9a-f]{32}-[0-9a-f]{16}-01\z/.match?(request.headers['Traceparent'].to_s) &&
        !request.headers['X-Request-Id'].to_s.empty?
    end
  end

  test "sends the versioned contract and service credential" do
    call = upstream.to_return(status: 200, body: '{"reply":"Python reply","tokens":0}')
    assert_equal "Python reply", @client.complete("Hello")
    assert_requested call, times: 1
  end

  test "assigns both timeouts and converts read timeout" do
    call = upstream.to_raise(Net::ReadTimeout)
    settings = []
    trace = TracePoint.new(:call) do |event|
      if event.method_id == :request && event.self.is_a?(Net::HTTP)
        http = event.self
        if http.address == "127.0.0.1" && http.port == 8000
          settings << [http.open_timeout, http.read_timeout, http.max_retries]
        end
      end
    end
    error = nil
    trace.enable do
      error = assert_raises(LlmClient::BoundaryError) { @client.complete("Hello") }
    end
    assert_not_empty settings, "Expected to observe Net::HTTP#request"
    assert settings.all? { |values| values == [2.0, 5.0, 0] }, settings.inspect
    assert_equal "ai_read_timeout", error.code
    assert_equal 504, error.status
    assert_requested call, times: 1
  end

  test "classifies transport and upstream failures" do
    [
      [Net::OpenTimeout, "ai_connect_timeout", 504],
      [Errno::ECONNREFUSED, "ai_connection_refused", 503]
    ].each do |exception, code, status|
      WebMock.reset!
      upstream.to_raise(exception)
      error = assert_raises(LlmClient::BoundaryError) { @client.complete("Hello") }
      assert_equal [code, status], [error.code, error.status]
    end
  end

  test "rejects bad statuses and malformed successful replies" do
    [
      [401, '{}', "ai_auth_rejected"],
      [422, '{}', "ai_contract_rejected"],
      [500, '{}', "ai_upstream_error"],
      [200, 'not json', "ai_invalid_response"],
      [200, '{"reply":42,"tokens":0}', "ai_invalid_response"],
      [200, '{"reply":"ok","tokens":-1}', "ai_invalid_response"]
    ].each do |status, body, code|
      WebMock.reset!
      upstream.to_return(status: status, body: body)
      error = assert_raises(LlmClient::BoundaryError) { @client.complete("Hello") }
      assert_equal [code, 502], [error.code, error.status]
    end
  end

  test "provider rate limit preserves safe retry metadata without a Rails retry" do
    call = upstream.to_return(status: 429, body: JSON.generate(detail: {
      code: "provider_rate_limited", retryable: true, retry_after: 2, attempts: 2
    }))
    error = assert_raises(LlmClient::BoundaryError) { @client.complete("Hello") }
    assert_equal ["provider_rate_limited", 429, true, 2],
      [error.code, error.status, error.retryable, error.retry_after]
    assert_requested call, times: 1
  end

  test "unsafe retry hints are omitted" do
    ["2\r\nInjected: yes", -1, 61, 1.5, nil].each do |raw|
      WebMock.reset!
      upstream.to_return(status: 429, body: JSON.generate(detail: {
        code: "provider_rate_limited", retryable: true, retry_after: raw
      }))
      error = assert_raises(LlmClient::BoundaryError) { @client.complete("Hello") }
      assert_nil error.retry_after
    end
  end

  test "provider authentication failure differs from service authentication failure" do
    upstream.to_return(status: 401, body: JSON.generate(detail: {
      code: "provider_auth_failed", retryable: false, retry_after: nil, attempts: 1
    }))
    error = assert_raises(LlmClient::BoundaryError) { @client.complete("Hello") }
    assert_equal "provider_auth_failed", error.code
    assert_equal 502, error.status
    assert_equal false, error.retryable
  end
end
