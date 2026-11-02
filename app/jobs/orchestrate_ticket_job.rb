class OrchestrateTicketJob < ApplicationJob
  queue_as :default

  # The caller is the existing authenticated ticket boundary. Never pass
  # customer-supplied tenant identifiers as an execution-context substitute.
  def perform(request, execution_context, request_id, traceparent, task_id: nil)
    if task_id
      task = AiTask.find_by!(id: task_id, operation: "agent")
      AiTaskDispatcher.run(task)
      return
    end
    raise ArgumentError, "missing execution context" if execution_context.to_s.empty?
    raise ArgumentError, "missing request identity" if request_id.to_s.empty?

    AgentOrchestrationClient.new.execute(
      request,
      execution_context: execution_context,
      request_id: request_id,
      traceparent: traceparent
    )
  end

  # Called only after the upload boundary persists sanitized evidence.
  def self.enqueue_attachment(organization_id, attachment_id)
    ProcessTicketAttachmentJob.perform_later(organization_id, attachment_id)
  end
end
