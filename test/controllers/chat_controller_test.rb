require "test_helper"

class ChatControllerTest < ActionDispatch::IntegrationTest
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
end
