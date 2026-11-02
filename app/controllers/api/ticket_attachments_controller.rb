require 'stringio'

class Api::TicketAttachmentsController < ApplicationController
  protect_from_forgery with: :exception
  before_action :authorize_ticket
  rescue_from TicketAccessPolicy::Unauthenticated, with: -> { head :unauthorized }
  rescue_from ActiveRecord::RecordNotFound, with: -> { head :not_found }

  def index
    @attachments = @ticket.ticket_attachments.where(organization_id: @ticket.organization_id).order(:id)
    render 'chat/show'
  end

  def create
    kind = params[:kind].to_s
    return head :unprocessable_entity unless %w[screenshot scanned_page voice].include?(kind)
    upload = params[:file]
    return head :unprocessable_entity unless upload.respond_to?(:read)
    limit = MultimodalClient.limit_for(kind)
    raw = upload.read(limit + 1)
    return head :payload_too_large if raw.bytesize > limit
    conversation = @ticket.conversation || @ticket.create_conversation!(organization: @ticket.organization)
    event = { adapter: { 'screenshot' => 'vision', 'scanned_page' => 'ocr', 'voice' => 'audio' }.fetch(kind),
      confidence: nil, gate_fired: true, tool_requested: false, tenant_id: @ticket.organization_id,
      conversation_id: conversation.id, request_id: request.request_id }
    item = nil
    ActiveSupport::Notifications.instrument('anchor.attachment.process', event) do
      result = begin
        MultimodalClient.new.extract(raw, kind: kind, tenant_id: @ticket.organization_id,
          conversation_id: conversation.id)
      rescue MultimodalClient::Unavailable
        { 'status' => 'error', 'evidence' => [], 'confidence' => nil,
          'notice' => 'Could not process attachment; re-upload or ask a human for help.' }
      end
      raw.clear
      status = result.fetch('status')
      raise MultimodalClient::Unavailable unless %w[readable unreadable error].include?(status)
      event[:confidence] = result['confidence']
      event[:gate_fired] = result['gate_fired'] == true || status != 'readable'
      TicketAttachment.transaction do
        item = @ticket.ticket_attachments.create!(organization: @ticket.organization,
          kind: kind, status: status == 'readable' ? 'queued' : status,
          notice: result['notice'], evidence: [], offline_fixture: result['offline_fixture'] == true)
        if status == 'readable'
          evidence = result.fetch('evidence').map { |row| row.merge('attachment_id' => item.id) }
          MultimodalClient.new.answer(evidence, tenant_id: @ticket.organization_id,
            conversation_id: conversation.id)
          item.update!(evidence: evidence)
          item.evidence_file.attach(io: StringIO.new(JSON.generate(evidence)),
            filename: "evidence-#{item.id}.json", content_type: 'application/json', identify: false)
        end
      end
      event[:attachment_id] = item.id
    end
    ProcessTicketAttachmentJob.perform_later(@ticket.organization_id, item.id) if item.status == 'queued'
    render json: representation(item), status: :accepted
  rescue MultimodalClient::Unavailable, KeyError, TypeError
    render json: { notice: 'Could not process attachment; please re-upload.' }, status: :unprocessable_entity
  ensure
    raw&.clear
  end

  def show
    item = @policy.attachment(@ticket.id, params[:id])
    render json: representation(item)
  end

  private

  def authorize_ticket
    @policy = TicketAccessPolicy.new(session)
    @ticket = @policy.ticket(params[:ticket_id])
  end

  def representation(item)
    { id: item.id, status: item.status, notice: item.notice,
      evidence: item.evidence, answer: item.answer, offline_fixture: item.offline_fixture }
  end
end
