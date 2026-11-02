require "test_helper"

class MultiAgentMilestoneTest < ActionDispatch::IntegrationTest
  include ActiveJob::TestHelper

  def product_adapter
    assert defined?(Anchor::MultiAgentReleaseAdapter),
      "Missing real Anchor composition: do not replace it with a standalone runner"
    Anchor::MultiAgentReleaseAdapter.new(self)
  end

  test "asynchronous ticket reaches cited retrieval handoff MCP and sandbox" do
    unless defined?(Anchor::MultiAgentReleaseAdapter)
      assert_enqueued_with(job: OrchestrateTicketJob, queue: "default") do
        OrchestrateTicketJob.perform_later({ "run_id" => "R-17" },
          "signed-context", "request-r17", "00-#{'a' * 32}-#{'b' * 16}-01")
      end
      error = assert_raises(AgentOrchestrationClient::Unavailable) do
        AgentOrchestrationClient.new(secret: nil).execute({},
          execution_context: "signed-context", request_id: "request-r17",
          traceparent: "00-#{'a' * 32}-#{'b' * 16}-01")
      end
      assert_equal "service_not_configured", error.message
      return
    end
    adapter = product_adapter
    result = adapter.run_ticket(
      tenant_id: 1,
      ticket_ref: "R-17",
      restart_at_specialist_handoff: true,
      message: "Check the delayed order and verify a policy-backed half-refund proposal. Do not issue a refund."
    )
    assert_equal true, result.fetch(:accepted_asynchronously)
    assert_equal true, result.fetch(:job_performed)
    assert result.fetch(:job_id).present?
    assert_equal 1, result.fetch(:tenant_id)
    assert_equal 0, result.fetch(:unapproved_mutations)

    citations = result.fetch(:citations)
    assert citations.any?, "No real seeded citations returned"
    citations.each do |citation|
      assert_equal true, citation.fetch(:resolved_from_seeded_store)
      assert_equal 1, citation.fetch(:tenant_id)
      assert citation.fetch(:chunk_id).present?
      assert citation.fetch(:content_sha256).present?
    end

    handoff = result.fetch(:specialist_handoff)
    assert_equal true, handoff.fetch(:persisted)
    assert_equal true, handoff.fetch(:resumed_after_restart)
    assert handoff.fetch(:context_digest_before).present?
    assert_equal handoff.fetch(:context_digest_before), handoff.fetch(:context_digest_after)
    refute_equal handoff.fetch(:from_role), handoff.fetch(:to_role)

    mcp = result.fetch(:mcp)
    assert_equal true, mcp.fetch(:local_server)
    [:initialized, :discovered, :called, :authorized].each do |key|
      assert_equal true, mcp.fetch(key), "Missing MCP evidence: #{key}"
    end

    sandbox = result.fetch(:sandbox)
    assert_equal true, sandbox.fetch(:service_execution)
    assert_equal true, sandbox.fetch(:isolated)
    assert_equal true, sandbox.fetch(:approved_script)
    assert_equal true, sandbox.fetch(:egress_denied)
    assert_equal true, sandbox.fetch(:limits_enforced)
    assert_equal "completed", sandbox.fetch(:exit_reason)
    assert sandbox.fetch(:execution_id).present?
    assert_equal [result.fetch(:trace_id)], result.fetch(:span_trace_ids).uniq
  end

  test "human only switch transfers full redacted context and stops dispatch" do
    unless defined?(Anchor::MultiAgentReleaseAdapter)
      calls = []
      transport = lambda do |_uri, request|
        calls << request
        Struct.new(:code, :body).new("503", '{"detail":"orchestration_not_configured"}')
      end
      client = AgentOrchestrationClient.new(secret: "test-secret", transport: transport)
      assert_raises(AgentOrchestrationClient::Unavailable) do
        client.execute({ "run_id" => "R-17" }, execution_context: "signed-context",
          request_id: "request-r17", traceparent: "00-#{'a' * 32}-#{'b' * 16}-01")
      end
      assert_equal 1, calls.length
      assert_equal "/v1/multi_agent", calls.first.path
      assert_equal "signed-context", calls.first["X-Anchor-Context"]
      assert_equal "request-r17", calls.first["X-Request-Id"]
      assert_equal({ "run_id" => "R-17" }, JSON.parse(calls.first.body))
      return
    end
    result = product_adapter.run_human_only_drill(tenant_id: 1, ticket_ref: "R-17")
    assert_equal 0, result.fetch(:post_switch_tool_calls)
    assert_equal true, result.fetch(:automation_stopped)
    handoff = result.fetch(:handoff)
    assert_equal true, handoff.fetch(:persisted)
    assert_equal true, handoff.fetch(:redacted)
    assert_equal 1, handoff.fetch(:tenant_id)
    [:ticket, :evidence, :completed_actions, :pending_actions, :approval_state,
     :trace_id, :orchestration_version, :reason].each do |key|
      assert handoff.key?(key), "Human handoff lost #{key}"
    end
    assert_equal result.fetch(:trace_id), handoff.fetch(:trace_id)
    assert handoff.fetch(:reason).present?
  end
end
