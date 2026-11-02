class ChatController < ApplicationController
  # This initial endpoint has no cookie-based authentication.
  # Revisit CSRF protection before adding session-cookie authentication.
  skip_forgery_protection

  def create
    context = AiTelemetry.context(request_id: request.request_id,
      traceparent: request.headers['traceparent'])
    response.set_header('X-Request-Id', context.fetch(:request_id))
    AiTelemetry.span('http.chat', context: context, stage: 'chat') do
      message = params.require(:message)
      reply = LlmClient.new.complete(message)
      render json: { reply: reply }
    end
  end
end
