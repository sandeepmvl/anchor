class AiTaskDispatcher
  class ReleaseBlocked < StandardError; end

  def self.release_check!(production: Rails.env.production?, evidence: ENV)
    return unless production
    unless evidence["ANCHOR_INGESTION_IDEMPOTENCY_VERIFIED"] == "1" &&
        evidence["ANCHOR_AGENT_IDEMPOTENCY_VERIFIED"] == "1"
      raise ReleaseBlocked, "downstream_idempotency_evidence_missing"
    end
    AiIncidentControl.release_check!(production: production, evidence: evidence)
    AiQuotaPolicy.release_check!(production: production,
      config: JSON.parse(Rails.root.join('config/ai_budgets.yml').read), evidence: evidence)
  end

  def self.submit_ingestion!(organization_id:, submission_key:, references:)
    release_check!
    task = AiTask.submit!(organization_id: organization_id, submission_key: submission_key,
      operation: "ingestion", payload: { "references" => references })
    deliver(task)
    task
  end

  # Only the authenticated ticket boundary may supply this execution context.
  def self.submit_agent!(organization_id:, submission_key:, request:, execution_context:, request_id:, traceparent:)
    release_check!
    task = AiTask.submit!(organization_id: organization_id, submission_key: submission_key,
      operation: "agent", payload: { "request" => request,
        "execution_context" => execution_context, "request_id" => request_id,
        "traceparent" => traceparent })
    deliver(task)
    task
  end

  def self.deliver(task)
    return if %w[complete dead_lettered].include?(task.state)
    if task.operation == "ingestion"
      IngestDocumentJob.perform_later(task.organization_id, nil, task_id: task.id)
    else
      OrchestrateTicketJob.perform_later(nil, nil, nil, nil, task_id: task.id)
    end
  rescue StandardError
    # Acceptance is durable even when the queue database is unavailable.
    # The recurring sweep retries delivery; never log payloads or credentials.
    Rails.logger.warn("ai_task_delivery_pending id=#{task.id}")
  end

  def self.sweep
    release_check!
    AiTask.eligible.order(:id).limit(100).each { |task| deliver(task) }
  end

  def self.run(task)
    release_check!
    ingestion = lambda do |tenant, reference, key|
      client = DocumentIngestionClient.new
      unless client.respond_to?(:ingest_idempotently)
        raise ReleaseBlocked, "ingestion_idempotency_adapter_required"
      end
      client.ingest_idempotently(tenant_id: tenant, reference: reference,
        idempotency_key: key)
    end
    agent = lambda do |payload, key|
      AiIncidentControl.run_agent(organization_id: task.organization_id,
        payload: payload, key: key) do
        AgentOrchestrationClient.new.execute(payload.fetch("request"),
          execution_context: payload.fetch("execution_context"),
          request_id: payload.fetch("request_id"), traceparent: payload["traceparent"],
          idempotency_key: key)
      end
    end
    task.run!(ingestion: ingestion, agent: agent)
  end
end
