class ProcessTicketAttachmentJob < ApplicationJob
  queue_as :default

  def perform(organization_id, attachment_id)
    item = TicketAttachment.where(organization_id: organization_id).find(attachment_id)
    item.with_lock do
      return unless item.status == 'queued'
      ticket = Ticket.where(organization_id: organization_id).find(item.ticket_id)
      conversation = Conversation.where(organization_id: organization_id, ticket_id: ticket.id).sole
      begin
        raise MultimodalClient::Unavailable unless item.evidence_file.attached?
        bytes = item.evidence_file.download
        raise MultimodalClient::Unavailable if bytes.bytesize > 65_536
        evidence = JSON.parse(bytes)
        raise MultimodalClient::Unavailable unless evidence == item.evidence
        answer = AgentOrchestrationClient.new.answer_attachments(evidence,
          tenant_id: organization_id, conversation_id: conversation.id)
        raise MultimodalClient::Unavailable unless answer['tool_calls'] == []
        item.update!(status: 'answered', answer: answer)
        conversation.update!(attachment_answer: answer)
      rescue StandardError
        item.update!(status: 'error', evidence: [], answer: {},
          notice: 'Could not process attachment; re-upload or ask a human for help.')
        item.evidence_file.purge if item.evidence_file.attached?
      end
    end
  end
end
