class IngestDocumentJob < ApplicationJob
  queue_as :default
  retry_on DocumentIngestionClient::Error, wait: 1.second, attempts: 3

  def perform(tenant_id, reference, task_id: nil)
    if task_id
      task = AiTask.find_by!(id: task_id, organization_id: tenant_id, operation: "ingestion")
      AiTaskDispatcher.run(task)
      return
    end
    unless tenant_id.is_a?(Integer) && tenant_id.positive? &&
           reference.is_a?(String) && reference.match?(/\A[0-9a-f]{32}\z/)
      raise ArgumentError, 'invalid_ingestion_reference'
    end
    DocumentIngestionClient.new.ingest(tenant_id: tenant_id, reference: reference)
  end
end
