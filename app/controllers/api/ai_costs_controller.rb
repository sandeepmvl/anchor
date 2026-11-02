module Api
  class AiCostsController < ActionController::Base
    def create
      expected = ENV['ANCHOR_COST_INGEST_SECRET'].to_s
      return head :service_unavailable if expected.empty?
      supplied = request.headers['X-Anchor-Cost-Secret'].to_s
      return head :unauthorized unless ActiveSupport::SecurityUtils.secure_compare(supplied, expected)
      return head :payload_too_large if request.raw_post.bytesize > 8192
      event = JSON.parse(request.raw_post)
      inserted = AiUsageEvent.ingest!(event)
      ActiveSupport::Notifications.instrument('anchor.cost.ingested',
        tenant_id: event.fetch('tenant_id'), inserted: inserted)
      render json: { inserted: inserted }
    rescue JSON::ParserError, ArgumentError, KeyError, TypeError, ActiveRecord::RecordNotFound
      head :unprocessable_entity
    end
  end
end
