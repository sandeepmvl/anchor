module AiQuotaEnforcement
  extend ActiveSupport::Concern

  included do
    before_action :enforce_ai_request_quota!, if: -> {
      %w[create cited].include?(action_name) &&
        (Rails.env.production? || request.headers['X-Anchor-Quota-Context'].present?)
    }
    before_action :check_ai_quota_release!, if: -> {
      Rails.env.production? && %w[create cited].include?(action_name)
    }
  end

  private

  def quota_policy
    AiQuotaPolicy.default
  end

  def quota_identity
    @quota_identity ||= AiQuotaPolicy.verify_context(
      request.headers['X-Anchor-Quota-Context'],
      secret: ENV['ANCHOR_QUOTA_SECRET'])
  end

  def enforce_ai_request_quota!
    quota_policy.rate!(quota_identity)
  rescue AiQuotaPolicy::Denied => error
    deny_ai_quota(error)
  end

  def check_ai_quota_release!
    AiQuotaPolicy.release_check!(production: true,
      config: JSON.parse(Rails.root.join('config/ai_budgets.yml').read), evidence: ENV)
  rescue AiQuotaPolicy::Denied => error
    deny_ai_quota(error)
  end

  def deny_ai_quota(error)
    response.set_header('Retry-After', error.retry_after.to_s) if error.retry_after
    render json: { error: error.code }, status: error.status
  end

  # Identity is not a reusable permission to spend.
  def quota_context_header
    request.headers['X-Anchor-Quota-Context']
  end
end
