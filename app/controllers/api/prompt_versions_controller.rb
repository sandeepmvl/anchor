module Api
  class PromptVersionsController < ApplicationController
    before_action :authenticate_internal_service!

    def active
      render json: PromptRegistry.active_reply
    end

    private

    def authenticate_internal_service!
      expected = ENV['ANCHOR_AI_SECRET']
      if expected.nil? || expected.empty?
        render json: { detail: 'service_not_configured' }, status: :service_unavailable
        return
      end
      supplied = request.headers['X-Anchor-Secret'].to_s
      unless ActiveSupport::SecurityUtils.secure_compare(supplied, expected)
        render json: { detail: 'service_auth_failed' }, status: :unauthorized
      end
    end
  end
end
