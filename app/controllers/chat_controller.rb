class ChatController < ApplicationController
  # This initial endpoint has no cookie-based authentication.
  # Revisit CSRF protection before adding session-cookie authentication.
  skip_forgery_protection

  def create
    message = params.require(:message)
    reply = LlmClient.new.complete(message)
    render json: { reply: reply }
  end
end
