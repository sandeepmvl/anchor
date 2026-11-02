class OrchestrateTicketJob < ApplicationJob
  queue_as :default

  # Queue wait is separate from retrieval/tool/LLM execution time. Do not log
  # request bodies, tenant credentials or customer-provided trace identities.
  around_perform do |job, block|
    enqueued = job.enqueued_at
    queue_wait_ms = enqueued ? [(Time.current - enqueued) * 1000, 0].max : nil
    ActiveSupport::Notifications.instrument(
      'anchor.ticket_worker',
      job_id: job.job_id, queue: job.queue_name, queue_wait_ms: queue_wait_ms
    ) { block.call }
  end

  # The caller is the existing authenticated ticket boundary. Never pass
  # customer-supplied tenant identifiers as an execution-context substitute.
  def perform(request, execution_context, request_id, traceparent, task_id: nil, tenant_credential: nil)
    if task_id
      task = AiTask.find_by!(id: task_id, operation: "agent")
      AiTaskDispatcher.run(task)
      return
    end
    raise ArgumentError, "missing execution context" if execution_context.to_s.empty?
    raise ArgumentError, "missing request identity" if request_id.to_s.empty?

    AgentOrchestrationClient.new.execute(
      PiiRedactionClient.value(request),
      execution_context: execution_context,
      request_id: request_id,
      traceparent: traceparent,
      tenant_credential: tenant_credential
    )
  end

  # Invoke from the trusted answer-persistence boundary, after committing the
  # exact answer version. Do not derive organization from customer JSON.
  def self.completed_answer(**attributes)
    sample = AnswerQualitySample.capture!(**attributes)
    ScoreAnswerQualityJob.perform_later(sample.organization_id, sample.id) if sample&.status == 'pending'
    sample
  end

  # Called only after the upload boundary persists sanitized evidence.
  def self.enqueue_attachment(organization_id, attachment_id)
    ProcessTicketAttachmentJob.perform_later(organization_id, attachment_id)
  end
end
