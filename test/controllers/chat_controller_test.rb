require "test_helper"
require "webmock/minitest"
require "securerandom"

WebMock.disable_net_connect!

class ChatControllerTest < ActionDispatch::IntegrationTest
  setup do
    @old_provider = Rails.configuration.x.llm[:provider]
    Rails.configuration.x.llm[:provider] = "stub"
    @keys = %w[ANCHOR_AI_SECRET ANCHOR_AI_URL ANCHOR_AI_OPEN_TIMEOUT ANCHOR_AI_READ_TIMEOUT]
    @previous = @keys.to_h { |key| [key, ENV[key]] }
    ENV["ANCHOR_AI_SECRET"] = SecureRandom.hex(24)
    ENV["ANCHOR_AI_URL"] = "http://127.0.0.1:8000/v1/infer"
    ENV["ANCHOR_AI_OPEN_TIMEOUT"] = "2"
    ENV["ANCHOR_AI_READ_TIMEOUT"] = "5"
  end

  teardown do
    Rails.configuration.x.llm[:provider] = @old_provider
    @previous.each { |key, value| value.nil? ? ENV.delete(key) : ENV[key] = value }
  end

  test "POST /chat keeps its JSON contract and delegates to LlmClient" do
    calls = []
    trace = TracePoint.new(:call) do |event|
      if event.defined_class == LlmClient && event.method_id == :complete
        calls << event.binding.local_variable_get(:message)
      end
    end

    trace.enable do
      post "/chat", params: { message: "Where is my order?" }, as: :json
    end

    assert_equal 200, response.status
    assert_equal "application/json", response.media_type
    assert_equal({ "reply" => LlmClient::CANNED_REPLY }, JSON.parse(response.body))
    assert_equal ["Where is my order?"], calls
  end

  test "POST /chat still requires a message" do
    post "/chat", params: {}, as: :json
    assert_response :bad_request
  end

  def upstream
    Rails.configuration.x.llm[:provider] = "http"
    stub_request(:post, ENV.fetch("ANCHOR_AI_URL")).with(
      headers: {
        "X-Anchor-Secret" => ENV.fetch("ANCHOR_AI_SECRET"),
        "Content-Type" => "application/json"
      },
      body: JSON.generate(
        version: "v1", conversation_id: 1,
        message: "Where is my order?", tenant_id: 1
      )
    )
  end

  def send_message
    post "/chat", params: { message: "Where is my order?" }, as: :json
  end

  test "HTTP runtime preserves the successful chat contract" do
    call = upstream.to_return(status: 200, body: '{"reply":"Python reply","tokens":0}')
    send_message
    assert_equal 200, response.status
    assert_equal "application/json", response.media_type
    assert_equal({ "reply" => "Python reply" }, JSON.parse(response.body))
    assert_requested call, times: 1
  end

  test "runtime timeout produces a bounded error response" do
    call = upstream.to_raise(Net::ReadTimeout)
    send_message
    assert_equal 504, response.status
    assert_equal "application/json", response.media_type
    assert_equal({ "error" => "ai_read_timeout" }, JSON.parse(response.body))
    assert_requested call, times: 1
  end

  test "Python auth rejection is not customer unauthorized" do
    call = upstream.to_return(status: 401, body: '{"detail":"service_auth_failed"}')
    send_message
    assert_equal 502, response.status
    assert_equal "application/json", response.media_type
    assert_equal({ "error" => "ai_auth_rejected" }, JSON.parse(response.body))
    assert_requested call, times: 1
  end
end
