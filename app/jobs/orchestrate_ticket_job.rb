class OrchestrateTicketJob < ApplicationJob
  queue_as :default

  # The caller is the existing authenticated ticket boundary. Never pass
  # customer-supplied tenant identifiers as an execution-context substitute.
  def perform(request, execution_context, request_id, traceparent)
    raise ArgumentError, "missing execution context" if execution_context.to_s.empty?
    raise ArgumentError, "missing request identity" if request_id.to_s.empty?

    AgentOrchestrationClient.new.execute(
      request,
      execution_context: execution_context,
      request_id: request_id,
      traceparent: traceparent
    )
  end
end
