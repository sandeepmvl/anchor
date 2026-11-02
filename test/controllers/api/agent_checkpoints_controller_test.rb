require "test_helper"
require "openssl"
require "base64"

class Api::AgentCheckpointsControllerTest < ActionDispatch::IntegrationTest
  setup do
    @old_secret = ENV["ANCHOR_APPROVAL_SECRET"]
    ENV["ANCHOR_APPROVAL_SECRET"] = "controller-test-secret-at-least-32-bytes"
    @checkpoint = AgentCheckpoint.create!(tenant_id: 1, conversation_id: 7,
      thread_id: "controller-refund-thread", action_key: "exact-refund-action",
      status: "pending", context_json: {
        order_id: "A100", amount_cents: 25000, classification: "large_refund",
        messages: ["Contact alex@example.test"], evidence: ["eligible"],
        decision_summary: "Call +1 202 555 0199", raw_secret: "never expose",
        pending_tool_call: { password: "never expose" }
      }.to_json)
    @base = "/api/agent_checkpoints/#{@checkpoint.thread_id}"
  end

  teardown do
    ENV["ANCHOR_APPROVAL_SECRET"] = @old_secret
  end

  def headers(tenant: 1, role: "refund_reviewer", actor: "reviewer-1", expiry: Time.now.to_i + 600)
    encoded = Base64.urlsafe_encode64({ tenant_id: tenant, role: role,
      actor_id: actor, exp: expiry }.to_json)
    signature = OpenSSL::HMAC.hexdigest("SHA256", ENV.fetch("ANCHOR_APPROVAL_SECRET"), encoded)
    { "Authorization" => "Bearer #{encoded}.#{signature}" }
  end

  test "reads require signed matching tenant and reviewer claims and redact state" do
    get @base
    assert_response :unauthorized
    get @base, headers: { "Authorization" => "Bearer forged.invalid" }
    assert_response :unauthorized
    get @base, headers: headers(expiry: Time.now.to_i - 1)
    assert_response :forbidden
    get @base, headers: headers(role: "customer")
    assert_response :forbidden
    get @base, headers: headers(tenant: 2)
    assert_response :not_found
    get @base, headers: headers
    assert_response :success
    assert_not_includes response.body, "alex@example.test"
    assert_not_includes response.body, "202 555"
    assert_not_includes response.body, "never expose"
    assert_not_includes response.body, "pending_tool_call"
    assert_includes response.body, "REDACTED_EMAIL"
    assert_equal "A100", response.parsed_body.fetch("context").fetch("order_id")
  end

  test "approval binds tenant actor and exact action and rejects replay" do
    assert_no_difference("AgentApproval.count") do
      post @base + "/approve", params: { action_key: @checkpoint.action_key }, as: :json
      assert_response :unauthorized
      post @base + "/approve", params: { action_key: @checkpoint.action_key },
        headers: headers(tenant: 2), as: :json
      assert_response :not_found
      post @base + "/approve", params: { action_key: @checkpoint.action_key },
        headers: headers(role: "customer"), as: :json
      assert_response :forbidden
      post @base + "/approve", params: { action_key: "different-action" },
        headers: headers, as: :json
      assert_response :conflict
    end
    assert_difference("AgentApproval.count", 1) do
      post @base + "/approve", params: { action_key: @checkpoint.action_key },
        headers: headers, as: :json
      assert_response :created
    end
    approval = AgentApproval.find(response.parsed_body.fetch("approval_id"))
    assert_equal 1, approval.tenant_id
    assert_equal "reviewer-1", approval.actor_id
    assert_equal @checkpoint.action_key, approval.action_key
    assert_equal "approved", @checkpoint.reload.status
    assert_no_difference("AgentApproval.count") do
      post @base + "/approve", params: { action_key: @checkpoint.action_key },
        headers: headers, as: :json
      assert_response :conflict
    end
  end
end
