require "openssl"
require "base64"
require "json"
require "securerandom"

class Api::AgentCheckpointsController < ApplicationController
  # This endpoint uses signed bearer credentials, not cookie authentication.
  skip_forgery_protection
  before_action :authenticate_reviewer!

  def show
    row = scoped_checkpoint
    return head :not_found unless row
    render json: { thread_id: row.thread_id, status: row.status,
                   action_key: row.action_key, context: safe_context(row) }
  end

  def approve
    row = scoped_checkpoint
    return head :not_found unless row
    return head :conflict unless params[:action_key] == row.action_key
    approval = nil
    AgentCheckpoint.transaction do
      changed = AgentCheckpoint.where(id: row.id, tenant_id: @claims["tenant_id"],
                                      status: "pending", action_key: params[:action_key])
                               .update_all(status: "approved", updated_at: Time.current)
      if changed == 1
        approval = AgentApproval.create!(agent_checkpoint: row,
          tenant_id: @claims["tenant_id"], actor_id: @claims["actor_id"],
          action_key: row.action_key, nonce: SecureRandom.uuid, status: "approved")
      end
    end
    return head :conflict unless approval
    render json: { approval_id: approval.id, thread_id: row.thread_id }, status: :created
  rescue ActiveRecord::RecordNotUnique
    head :conflict
  end

  private

  def authenticate_reviewer!
    secret = ENV["ANCHOR_APPROVAL_SECRET"]
    token = request.authorization.to_s.delete_prefix("Bearer ")
    encoded, signature = token.split(".", 2)
    unless secret && secret.bytesize >= 32 && encoded && signature
      return head :unauthorized
    end
    expected = OpenSSL::HMAC.hexdigest("SHA256", secret, encoded)
    unless signature.bytesize == expected.bytesize &&
           ActiveSupport::SecurityUtils.secure_compare(signature, expected)
      return head :unauthorized
    end
    @claims = JSON.parse(Base64.urlsafe_decode64(encoded))
    valid = @claims["tenant_id"].is_a?(Integer) && @claims["tenant_id"].positive? &&
      @claims["actor_id"].is_a?(String) && !@claims["actor_id"].empty? &&
      @claims["role"] == "refund_reviewer" && @claims["exp"].is_a?(Integer) &&
      @claims["exp"] > Time.now.to_i
    head :forbidden unless valid
  rescue ArgumentError, JSON::ParserError, TypeError
    head :unauthorized
  end

  def scoped_checkpoint
    AgentCheckpoint.find_by(thread_id: params[:thread_id], tenant_id: @claims["tenant_id"])
  end

  def safe_text(value)
    value.to_s.gsub(/[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}/, "[REDACTED_EMAIL]")
         .gsub(/\+?\d[\d ()-]{8,}\d/, "[REDACTED_PHONE]")
  end

  def safe_context(row)
    # Defense in depth: never return arbitrary persisted graph state.
    source = JSON.parse(row.context_json)
    result = {}
    %w[order_id amount_cents classification remaining_budget].each do |key|
      value = source[key]
      result[key] = value if value.is_a?(Integer) ||
        (value.is_a?(String) && value.match?(/\A[A-Za-z0-9_:-]{1,80}\z/))
    end
    result["decision_summary"] = safe_text(source["decision_summary"])
    result["messages"] = Array(source["messages"]).map { |v| safe_text(v) }
    result["evidence"] = Array(source["evidence"]).map { |v| safe_text(v) }
    result
  rescue JSON::ParserError
    {}
  end
end
