require "test_helper"
require "webmock/minitest"
require "securerandom"
require_relative '../../labs/ai-product-ux/test_helper'

WebMock.disable_net_connect!

class ChatControllerTest < ActionDispatch::IntegrationTest
  include UXFixtures

  setup do
    @old_provider = Rails.configuration.x.llm[:provider]
    Rails.configuration.x.llm[:provider] = "stub"
    @keys = %w[ANCHOR_AI_SECRET ANCHOR_AI_URL ANCHOR_AI_OPEN_TIMEOUT ANCHOR_AI_READ_TIMEOUT
      ANCHOR_CHAT_PROMPT_VERSION]
    @previous = @keys.to_h { |key| [key, ENV[key]] }
    ENV["ANCHOR_AI_SECRET"] = SecureRandom.hex(24)
    ENV["ANCHOR_AI_URL"] = "http://127.0.0.1:8000/v1/infer"
    ENV["ANCHOR_AI_OPEN_TIMEOUT"] = "2"
    ENV["ANCHOR_AI_READ_TIMEOUT"] = "5"
    ENV["ANCHOR_CHAT_PROMPT_VERSION"] = "chat:v1"
    @old_answer_generator = ChatController.answer_generator
  end

  teardown do
    Rails.configuration.x.llm[:provider] = @old_provider
    @previous.each { |key, value| value.nil? ? ENV.delete(key) : ENV[key] = value }
    ChatController.answer_generator = @old_answer_generator
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
      body: JSON.parse(JSON.generate(
        version: "v1", conversation_id: 1,
        message: "Where is my order?", tenant_id: 1,
        prompt_contract: PromptRegistry::V1.stringify_keys.merge("version" => "chat:v1")
      ))
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

  test "optional history and summary state cross the existing HTTP boundary" do
    Rails.configuration.x.llm[:provider] = "http"
    history = [{ "position" => 3, "role" => "user", "content" => "Receipt update" }]
    state = { "summary" => "Mira promised a receipt Friday.",
      "prompt_version" => "summary:v1", "folded_position" => 2 }
    expected = { "version" => "v1", "conversation_id" => 1, "tenant_id" => 1,
      "message" => "Where is my order?", "history" => history, "summary_state" => state,
      "prompt_contract" => PromptRegistry::V1.stringify_keys.merge("version" => "chat:v1") }
    call = stub_request(:post, ENV.fetch("ANCHOR_AI_URL")).with(
      headers: { "X-Anchor-Secret" => ENV.fetch("ANCHOR_AI_SECRET"),
        "Content-Type" => "application/json" }, body: expected
    ).to_return(status: 200, body: JSON.generate(
      reply: "Python reply", tokens: 12, context_state: state))
    post "/chat", params: { message: "Where is my order?", history: history,
      summary_state: state }, as: :json
    assert_response :success
    assert_equal({ "reply" => "Python reply", "context_state" => state }, JSON.parse(response.body))
    assert_requested call, times: 1
  end

  test 'cited chat returns quoted sources and uncertainty with verified scope' do
    ChatController.answer_generator = lambda do |question:, tenant_id:, conversation_id:|
      assert_equal 'When will my refund arrive?', question
      assert_equal 1, tenant_id
      assert_equal 'conversation-a', conversation_id
      {'prompt_version' => 'refund-ux:v1', 'retrieved_chunk_ids' => [31], 'cited_answer' => {
        'answer' => 'I searched the support knowledge base. Refunds require review. Check your receipt.',
        'decision' => 'hedge', 'citations' => [{'chunk_id' => 31, 'quoted_span' => 'Refunds require review.'}]}}
    end
    post '/chat/cited', params: {message: 'When will my refund arrive?', tenant_id: 999},
      headers: {'Authorization' => "Bearer #{token}"}, as: :json
    assert_response :success
    body = JSON.parse(response.body)
    assert_equal body['reply'], body['cited_answer']['answer']
    assert_equal 'hedge', body['cited_answer']['decision']
    assert_equal [31], body['citations']
    assert_equal false, body['abstained']
    assert_equal '1', GoldenCandidate::AnswerMessage.find(body['message_id']).tenant_id
    get '/stream_chat'
    assert_response :success
    assert_includes response.body, "item.textContent = 'Chunk '"
    assert_includes response.body, 'Uncertain — check conditions'
    assert_includes response.body, "'Accept': 'text/event-stream'"
    assert_includes response.body, 'reply.content !== answer.textContent'
  end

  test 'cited chat fails closed without generator configuration' do
    ChatController.answer_generator = nil
    assert_no_difference('GoldenCandidate::AnswerMessage.count') do
      post '/chat/cited', params: {message: 'Refund?'},
        headers: {'Authorization' => "Bearer #{token}"}, as: :json
      assert_response :service_unavailable
    end
  end
end
