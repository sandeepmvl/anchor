class Api::AiBudgetReservationsController < ActionController::API
  before_action :authenticate_service!
  rescue_from AiQuotaPolicy::Denied, with: :deny_quota

  def create
    render json: policy.reserve!(identity, bound: params.require(:bound),
      attempt_id: params.require(:attempt_id))
  end

  def update
    render json: policy.settle!(identity, reservation_id: params.require(:id),
      actual: params.require(:actual))
  end

  def index
    render json: policy.snapshot(identity)
  end

  private

  def policy
    @policy ||= AiQuotaPolicy.default
  end

  def identity
    @identity ||= AiQuotaPolicy.verify_context(request.headers['X-Anchor-Quota-Context'],
      secret: ENV['ANCHOR_QUOTA_SECRET'])
  end

  def authenticate_service!
    expected = ENV['ANCHOR_AI_SECRET'].to_s
    received = request.headers['X-Anchor-Secret'].to_s
    if expected.empty?
      raise AiQuotaPolicy::Denied.new('quota_authority_unavailable', 503)
    end
    unless expected.bytesize == received.bytesize &&
        ActiveSupport::SecurityUtils.secure_compare(expected, received)
      raise AiQuotaPolicy::Denied.new('quota_service_auth_failed', 401)
    end
  end

  def deny_quota(error)
    response.set_header('Retry-After', error.retry_after.to_s) if error.retry_after
    render json: { error: error.code }, status: error.status
  end
end
