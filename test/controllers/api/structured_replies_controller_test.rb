require "test_helper"
require "net/http"
require "json"

class Api::StructuredRepliesControllerTest < ActionDispatch::IntegrationTest
  class FakeHttp
    attr_accessor :use_ssl, :open_timeout, :read_timeout, :max_retries
    attr_reader :requests

    def initialize(status, body, failure: nil)
      @status, @body, @failure = status, body, failure
      @requests = []
    end

    def request(request)
      @requests << request
      raise @failure if @failure
      Struct.new(:code, :body).new(@status.to_s, JSON.generate(@body))
    end
  end

  def with_runtime(status, body, failure: nil)
    transport = FakeHttp.new(status, body, failure: failure)
    original = Net::HTTP.method(:new)
    old_secret = ENV["ANCHOR_AI_SECRET"]
    ENV["ANCHOR_AI_SECRET"] = "rails-test-secret"
    Net::HTTP.define_singleton_method(:new) { |*args, **kwargs| transport }
    yield transport
  ensure
    Net::HTTP.define_singleton_method(:new, original) if original
    old_secret.nil? ? ENV.delete("ANCHOR_AI_SECRET") : ENV["ANCHOR_AI_SECRET"] = old_secret
  end

  def submit
    post "/api/structured_replies", params: { message: "Checkout is down for every customer; help!" }, as: :json
  end

  test "returns validated triage and uses authenticated runtime envelope" do
    with_runtime(200, { category: "technical", urgency: "P1" }) do |http|
      submit
      assert_response :success
      assert_equal({ "category" => "technical", "urgency" => "P1" }, response.parsed_body)
      request = http.requests.fetch(0)
      assert request.path.end_with?("/structured_replies")
      assert_equal "rails-test-secret", request["X-Anchor-Secret"]
      assert request["X-Request-Id"].present?
      assert_equal({ "version" => "v1", "tenant_id" => 1, "conversation_id" => 1,
        "message" => "Checkout is down for every customer; help!" }, JSON.parse(request.body))
      assert_equal 0, http.max_retries
    end
  end

  test "decode errors contain no field errors or raw provider content" do
    with_runtime(422, { detail: { code: "invalid_model_output", reason: "invalid_json",
      raw_excerpt: "PRIVATE PROVIDER RESPONSE", errors: [{ input: "PRIVATE" }] } }) do
      submit
      assert_response :unprocessable_entity
      assert_equal({ "error" => "invalid_model_output", "reason" => "invalid_json" }, response.parsed_body)
      assert_not_includes response.body, "PRIVATE"
    end
  end

  test "schema errors expose only safe field locations and types" do
    with_runtime(422, { detail: { code: "invalid_model_output", reason: "schema_violation",
      errors: [{ loc: ["urgency"], type: "missing", input: "PRIVATE", msg: "PRIVATE" }] } }) do
      submit
      assert_response :unprocessable_entity
      assert_equal({ "error" => "invalid_model_output", "reason" => "schema_violation",
        "errors" => [{ "loc" => ["urgency"], "type" => "missing" }] }, response.parsed_body)
      assert_not_includes response.body, "PRIVATE"
    end
  end

  test "generic runtime validation is not mislabeled as invalid model output" do
    with_runtime(422, { detail: "validation_failed" }) do
      submit
      assert_response :bad_gateway
      assert_equal "ai_contract_rejected", response.parsed_body["error"]
    end
  end

  test "rejects invalid successful runtime payload" do
    with_runtime(200, { category: "technical", urgency: "P1", raw: "PRIVATE" }) do
      submit
      assert_response :bad_gateway
      assert_equal({ "error" => "ai_invalid_response" }, response.parsed_body)
    end
  end

  test "preserves timeout mapping" do
    with_runtime(200, {}, failure: Net::ReadTimeout.new) do
      submit
      assert_response :gateway_timeout
      assert_equal "ai_read_timeout", response.parsed_body["error"]
    end
  end

  test "rejects blank message before contacting runtime" do
    with_runtime(200, {}) do |http|
      post "/api/structured_replies", params: { message: " " }, as: :json
      assert_response :unprocessable_entity
      assert_empty http.requests
    end
  end
end
