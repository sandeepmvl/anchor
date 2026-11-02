require "test_helper"
require "socket"

class LlmClientTest < ActiveSupport::TestCase
  test "default client answers without a key or external connection" do
    original_key = ENV.delete("OPENAI_API_KEY")
    assert_equal "stub", Rails.configuration.x.llm.fetch(:provider)
    network_calls = %i[connect connect_nonblock tcp open getaddrinfo]
    guard = TracePoint.new(:call, :c_call) do |event|
      receiver = event.self
      socket_receiver = BasicSocket === receiver ||
        (Class === receiver && receiver <= BasicSocket)
      if (socket_receiver || Addrinfo.equal?(receiver) || Socket.equal?(receiver)) &&
          network_calls.include?(event.method_id)
        raise "Offline client attempted network access"
      end
    end

    guard.enable do
      client = LlmClient.new
      assert_equal LlmClient::CANNED_REPLY, client.complete("Where is my order?")
      assert_equal LlmClient::CANNED_REPLY, client.complete("Can I return it?")
    end
  ensure
    if original_key
      ENV["OPENAI_API_KEY"] = original_key
    end
  end
end
