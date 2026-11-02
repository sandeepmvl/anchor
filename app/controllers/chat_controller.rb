class ChatController < ApplicationController
  include ActionController::Live
  include AiQuotaEnforcement
  protect_from_forgery with: :exception
  # Only the original non-streaming JSON endpoint is exempt from CSRF.
  skip_forgery_protection if: -> { action_name == 'create' && !stream_request? }
  class_attribute :answer_generator, default: nil
  around_action :profile_anchor_action
  before_action :reject_untrusted_tenant_state, only: [:create, :cited]
  before_action :redact_chat_parameters, only: [:create, :cited]
  rescue_from PiiRedactionClient::Unavailable, with: :pii_unavailable

  class ProtocolError < StandardError; end

  def show
    return unless params[:ticket_id].present?
    policy = TicketAccessPolicy.new(session)
    @ticket = policy.ticket(params[:ticket_id])
    @human_only = AiIncidentControl.human_only?(@ticket.organization_id)
    @attachments = @ticket.ticket_attachments.where(organization_id: @ticket.organization_id).order(:id)
  rescue TicketAccessPolicy::Unauthenticated
    head :unauthorized
  rescue ActiveRecord::RecordNotFound
    head :not_found
  end

  def stream_request?
    request.headers['Accept'].to_s.split(',').any? do |value|
      value.strip.start_with?('text/event-stream')
    end
  end

  def create
    context = AiTelemetry.current&.child_context || AiTelemetry.context(request_id: request.request_id,
      traceparent: request.headers['traceparent'])
    response.set_header('X-Request-Id', context.fetch(:request_id))
    AiTelemetry.span('http.chat', context: context, stage: 'chat') do
      message = params.require(:message)
      persist_pii_ingestion(message)
      if @pii_conversation && AiIncidentControl.human_only?(@pii_conversation.organization_id)
        return render_incident_handoff('human_only')
      end
      # Python screens extracted KB evidence and customer text before generation.
      # Never promote message, history, or evidence into trusted approval context.
      retrieval = retrieval_settings
      if retrieval
        if stream_request?
          render json: { error: 'retrieval_stream_not_supported' }, status: :unprocessable_entity
          return
        end
        trusted, settings = retrieval
        result = AiTelemetry.measure('retrieval') do
          LlmClient.new.complete_retrieval(message, context: trusted,
            url: settings.fetch('url'), secret: ENV['ANCHOR_AI_SECRET'])
        end
        result = PiiRedactionClient.value(result)
        persist_pii_reply(result)
        render json: result
        return
      end
      config = YAML.safe_load(ERB.new(Rails.root.join('config/llm.yml').read).result).fetch('shared')
      selector = params.fetch(:prompt_version, config.fetch('default_prompt_version'))
      contract = PromptRegistry.resolve(selector)
      unless stream_request?
        Rails.logger.info(event: 'prompt_resolved', request_id: context.fetch(:request_id),
          prompt_version: contract.fetch('version'))
      end
      message = PromptRegistry::Message.new(message, contract)
      if params.key?(:history) || params.key?(:summary_state)
        options = params.permit(history: [:position, :role, :content],
          summary_state: [:summary, :prompt_version, :folded_position]).to_h
        message = LlmClient::ContextMessage.new(message, contract, options)
      end
      unless stream_request?
        reply = AiTelemetry.measure('model') { LlmClient.new.complete(message) }
        body = { reply: reply.to_s }
        body[:context_state] = reply.context_state if reply.respond_to?(:context_state)
        body = PiiRedactionClient.value(body)
        persist_pii_reply(body)
        render json: body
        return
      end
      response.headers['Content-Type'] = 'text/event-stream; charset=utf-8'
      response.headers['Cache-Control'] = 'no-cache'
      response.headers['X-Accel-Buffering'] = 'no'
      ActiveRecord::Base.connection_pool.release_connection
      write_stream_packet("event: status\ndata: Preparing your reply\n\n")
      relay(message)
      Rails.logger.info(event: 'prompt_resolved', request_id: context.fetch(:request_id),
        prompt_version: contract.fetch('version'))
    end
  rescue LlmClient::BoundaryError => error
    # Preserve the existing boundary contract without depending on lazy load-hook timing.
    response.set_header('Retry-After', error.retry_after.to_s) unless error.retry_after.nil?
    if %w[routing_unavailable routing_deadline routing_attempt_limit incident_human_only].include?(error.code)
      return render_incident_handoff(error.code)
    end
    body = { error: error.code }
    body.merge!(error.details) if error.code == 'invalid_model_output'
    if error.code.start_with?('provider_')
      body.merge!(retryable: error.retryable, retry_after: error.retry_after)
    end
    render json: body, status: error.status
  rescue ActiveRecord::RecordNotFound
    render json: { error: 'unknown_prompt_version' }, status: :unprocessable_entity
  ensure
    response.stream.close if stream_request?
  end

  def cited
    token = session[:anchor_customer_context].presence || request.authorization.to_s.delete_prefix('Bearer ')
    claims = GoldenCandidate.claims!(token, purpose: 'answer_customer')
    if AiIncidentControl.human_only?(claims.fetch('tenant_id').to_i)
      return render json: AiIncidentControl.handoff(
        { 'conversation_id' => claims.fetch('conversation_id'), 'message' => params[:message].to_s })
    end
    unless answer_generator.respond_to?(:call)
      return render json: {error: 'generation_unavailable'}, status: :service_unavailable
    end
    question = params.require(:message).to_s
    raise GoldenCandidate::InvalidInput if question.empty? || question.bytesize > 1800
    result = answer_generator.call(question: question,
      tenant_id: claims.fetch('tenant_id').to_i,
      conversation_id: claims.fetch('conversation_id'))
    result = PiiRedactionClient.value(result)
    answer = result.fetch('cited_answer')
    ids = result.fetch('retrieved_chunk_ids')
    version = result.fetch('prompt_version')
    unless answer.is_a?(Hash) && %w[answer hedge refuse].include?(answer['decision']) &&
        answer['answer'].is_a?(String) && answer['answer'].length.between?(1, 4000) &&
        answer['citations'].is_a?(Array) && ids.is_a?(Array) &&
        ids.all? { |id| id.is_a?(Integer) && id.positive? } &&
        version.is_a?(String) && version.match?(/\A[a-zA-Z0-9_:.-]{1,100}\z/) &&
        answer['citations'].all? { |c| c.is_a?(Hash) && ids.include?(c['chunk_id']) &&
          c['quoted_span'].is_a?(String) && c['quoted_span'].present? } &&
        (answer['decision'] == 'refuse' ? answer['citations'].empty? : answer['citations'].any?)
      raise GoldenCandidate::InvalidInput
    end
    safe = GoldenCandidate.safe!({'question' => question, 'cited_answer' => answer})
    stored = GoldenCandidate::AnswerMessage.create!(tenant_id: claims.fetch('tenant_id').to_s,
      customer_id: claims.fetch('subject'), conversation_id: claims.fetch('conversation_id'),
      prompt_version: version, retrieved_chunk_ids: ids, payload: safe)
    render json: {reply: answer.fetch('answer'), citations: answer.fetch('citations').map { |c| c.fetch('chunk_id') },
      abstained: answer.fetch('decision') == 'refuse', cited_answer: answer, message_id: stored.id}
  rescue GoldenCandidate::AuthorizationError
    head :unauthorized
  rescue GoldenCandidate::InvalidInput, KeyError, TypeError
    render json: {error: 'answer_rejected'}, status: :unprocessable_entity
  end

  private

  def render_incident_handoff(reason)
    context = { 'message' => params[:message].to_s }
    if @pii_conversation
      context['conversation_id'] = @pii_conversation.id
      context['ticket_id'] = @pii_conversation.ticket_id
    end
    body = AiIncidentControl.handoff(context, reason: reason)
    persist_pii_reply(body)
    if stream_request?
      response.headers['Content-Type'] = 'text/event-stream; charset=utf-8'
      payload = JSON.generate(code: reason, message: body.fetch('reply'), handoff: true)
      write_stream_packet("event: error\ndata: #{payload}\n\n")
    else
      render json: body
    end
  end

  def reject_untrusted_tenant_state
    authority = %w[approval confirmed execution_context tenant_credential role]
    if authority.any? { |key| params.key?(key) }
      render json: {error: 'untrusted_authorization_fields'}, status: :forbidden
      return
    end
    return unless Rails.env.production? || ENV.key?('ANCHOR_TENANT_CREDENTIALS')
    forbidden = %w[tenant_id organization_id conversation_id history summary_state checkpoint_id]
    if forbidden.any? { |key| params.key?(key) }
      render json: {error: 'untrusted_tenant_state'}, status: :forbidden
    end
  end

  def redact_chat_parameters
    selected = params.permit!.to_h.slice('message', 'history', 'summary_state', 'context')
    safe = PiiRedactionClient.value(selected)
    safe.each { |key, value| params[key] = value }
    params.delete(:notes)
    params.delete(:private_notes)
  end

  def persist_pii_ingestion(message)
    return unless params[:ticket_id].present?
    ticket = TicketAccessPolicy.new(session).ticket(params[:ticket_id])
    @pii_conversation = Conversation.create!(organization_id: ticket.organization_id,
      ticket_id: ticket.id, pii_payload: {'ingestion' => message.to_s})
    if Rails.env.production? || ENV.key?('ANCHOR_TENANT_CREDENTIALS')
      request.env['anchor.trusted_retrieval_context'] = RetrievalClient::TrustedContext.new(
        tenant_id: ticket.organization_id, conversation_id: @pii_conversation.id)
    end
  rescue TicketAccessPolicy::Unauthenticated
    head :unauthorized
    throw :abort
  end

  def persist_pii_reply(value)
    return unless @pii_conversation
    @pii_conversation.update!(pii_payload: @pii_conversation.pii_payload.merge('response' => value))
  end

  def pii_unavailable
    if stream_request? && response.committed?
      write_stream_error('pii_redaction_unavailable')
    else
      render json: {error: 'pii_redaction_unavailable'}, status: :service_unavailable
    end
  end

  def profile_anchor_action
    context = AiTelemetry.context(request_id: request.request_id,
      traceparent: request.headers['traceparent'])
    AiTelemetry.span('rails.action', context: context, stage: 'application') { yield }
  end

  def retrieval_settings
    settings = YAML.safe_load(Rails.root.join('config/retrieval.yml').read).fetch('chat')
    fixture = !Rails.env.production? && ENV[settings.fetch('fixture_env')] == '1'
    trusted = request.env['anchor.trusted_retrieval_context']
    if fixture
      trusted = RetrievalClient::TrustedContext.new(
        tenant_id: settings.fetch('fixture_tenant_id'),
        conversation_id: settings.fetch('fixture_conversation_id'))
    end
    enabled = settings.fetch('enabled_tenant_ids')
    unless fixture
      if trusted.nil?
        if enabled.any?
          raise LlmClient::BoundaryError.new('trusted_context_required', 422)
        end
        return nil
      end
      unless trusted.is_a?(RetrievalClient::TrustedContext)
        raise LlmClient::BoundaryError.new('trusted_context_required', 422)
      end
      return nil unless enabled.include?(trusted.tenant_id)
    end
    [trusted, settings]
  end

  def relay(message)
    buffer = +''.b
    text = +''
    terminal = false
    LlmClient.new.stream(message) do |chunk|
      buffer << chunk.b
      raise ProtocolError if buffer.bytesize > 1_048_576
      while (boundary = buffer.match(/\r?\n\r?\n/))
        packet = buffer.slice!(0, boundary.end(0)).force_encoding(Encoding::UTF_8)
        raise ProtocolError unless packet.valid_encoding?
        if packet.start_with?(':')
          write_stream_packet(packet)
          next
        end
        lines = packet.split(/\r?\n/)
        name = lines.find { |line| line.start_with?('event:') }&.delete_prefix('event:')&.strip
        data = lines.select { |line| line.start_with?('data:') }
          .map { |line| line.delete_prefix('data:').delete_prefix(' ') }.join("\n")
        case name
        when 'token'
          text << data
          raise ProtocolError if text.length > 65_536
        when 'final'
          payload = JSON.parse(data)
          raise ProtocolError unless payload.is_a?(Hash) && payload['content'] == text &&
            payload['citations'].is_a?(Array) && payload['citations'].all? { |c| c.is_a?(String) } &&
            [true, false].include?(payload['truncated'])
          payload = PiiRedactionClient.value(payload)
          safe_text = payload.fetch('content')
          persist_pii_reply(payload)
          encoded = safe_text.split("\n", -1).map { |line| "data: #{line}\n" }.join
          write_stream_packet("event: token\n#{encoded}\n", token: !safe_text.empty?)
          write_stream_packet("event: final\ndata: #{JSON.generate(payload)}\n\n")
          terminal = true
        when 'error'
          Rails.logger.warn(event: 'runtime_error_frame')
          write_stream_error('upstream_failed')
          terminal = true
        else
          raise ProtocolError
        end
        break if terminal
      end
      break if terminal
    end
    write_stream_error('early_end') unless terminal
  rescue IOError, Errno::EPIPE, Errno::ECONNRESET, ActionController::Live::ClientDisconnected
    Rails.logger.info(event: 'client_disconnect')
  rescue StandardError => error
    code = error.is_a?(LlmClient::BoundaryError) && error.code == 'ai_read_timeout' ? 'read_timeout' : 'runtime_failure'
    Rails.logger.warn(event: code, error_type: error.class.name)
    begin
      write_stream_error(code)
    rescue IOError, Errno::EPIPE, Errno::ECONNRESET, ActionController::Live::ClientDisconnected
      Rails.logger.info(event: 'client_disconnect')
    end
  end

  def write_stream_packet(packet, token: false)
    response.stream.write(packet)
    AiTelemetry.current&.mark_stream_write(token: token)
  end

  def write_stream_error(code)
    incident = %w[upstream_failed early_end read_timeout runtime_failure].include?(code)
    message = incident ? AiIncidentControl.config.fetch('message') : 'Reply interrupted; please retry.'
    payload = JSON.generate(code: code, message: message, handoff: incident)
    write_stream_packet("event: error\ndata: #{payload}\n\n")
  end
end
