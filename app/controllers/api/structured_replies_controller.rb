module Api
  class StructuredRepliesController < ApplicationController
    def create
      message = params[:message]
      unless message.is_a?(String) && message.length.between?(1, 4000) && !message.strip.empty?
        return render json: { error: "invalid_message" }, status: :unprocessable_entity
      end

      render json: LlmClient.new.structured_reply(message)
    end
  end
end
