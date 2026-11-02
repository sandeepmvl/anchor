require "json"
require "digest"
require "securerandom"

class AiTask < ApplicationRecord
  MAX_ATTEMPTS = 3
  LEASE_SECONDS = 90
  class Conflict < StandardError; end
  class LostClaim < StandardError; end

  def self.submit!(organization_id:, submission_key:, operation:, payload:)
    raise ArgumentError, "invalid_submission_key" unless
      submission_key.is_a?(String) && submission_key.match?(/\A[a-zA-Z0-9_-]{1,80}\z/)
    raise ArgumentError, "invalid_organization" unless organization_id.is_a?(Integer) && organization_id.positive?
    case operation
    when "ingestion"
      refs = payload.fetch("references")
      raise ArgumentError, "invalid_references" unless refs.is_a?(Array) && refs.length.between?(1, 1000) &&
        refs.uniq == refs && refs.all? { |r| r.is_a?(String) && r.match?(/\A[0-9a-f]{32}\z/) }
      raise ArgumentError, "invalid_payload" unless payload.keys == ["references"]
      total = refs.length
    when "agent"
      raise ArgumentError, "invalid_agent_payload" unless payload.keys.sort == %w[execution_context request request_id traceparent] &&
        payload["request"].is_a?(Hash) && payload["execution_context"].is_a?(String) &&
        !payload["execution_context"].empty? && !payload["request_id"].to_s.empty?
      total = 1
    else
      raise ArgumentError, "invalid_operation"
    end
    encoded = JSON.generate(payload)
    raise ArgumentError, "payload_too_large" if encoded.bytesize > 262144
    digest = Digest::SHA256.hexdigest(JSON.generate([operation, payload]))
    task = create_or_find_by!(organization_id: organization_id, submission_key: submission_key) do |row|
      row.operation = operation
      row.payload_json = encoded
      row.payload_digest = digest
      row.total = total
      row.available_at = Time.current
    end
    raise Conflict, "submission_key_conflict" unless task.payload_digest == digest
    task
  end

  def unit_key(index)
    "anchor:#{organization_id}:ai-task:#{id}:unit:#{index}"
  end

  def public_status
    { id: id, state: state, progress: progress, total: total,
      attempts: attempts, next_attempt_at: available_at,
      updated_at: updated_at, error: error_code }
  end

  def self.eligible(now = Time.current)
    where(state: %w[queued retrying]).where("available_at <= ?", now)
      .or(where(state: "running").where("lease_until <= ?", now))
  end

  def run!(ingestion:, agent:, now: Time.current, after_unit: nil)
    token = SecureRandom.hex(16)
    scope = self.class.eligible(now).where(id: id)
    if reload.attempts >= MAX_ATTEMPTS
      scope.update_all(state: "dead_lettered", error_code: "task_failed",
        claim_token: nil, lease_until: nil, updated_at: now)
      return
    end
    claimed = scope.where(attempts: attempts).update_all(state: "running",
      attempts: attempts + 1, claim_token: token, lease_until: now + LEASE_SECONDS,
      error_code: nil, updated_at: now)
    return unless claimed == 1
    reload
    owned = self.class.where(id: id, state: "running", claim_token: token)
    begin
      payload = JSON.parse(payload_json)
      while progress < total
        index = progress
        key = unit_key(index)
        if operation == "ingestion"
          ingestion.call(organization_id, payload.fetch("references").fetch(index), key)
        else
          agent.call(payload, key)
        end
        # Tests kill the worker here, after an effect but before its checkpoint.
        after_unit&.call(index)
        changed = owned.where(progress: index).update_all(progress: index + 1,
          lease_until: Time.current + LEASE_SECONDS, updated_at: Time.current)
        raise LostClaim unless changed == 1
        self.progress = index + 1
      end
      raise LostClaim unless owned.update_all(state: "complete", claim_token: nil,
        lease_until: nil, error_code: nil, updated_at: Time.current) == 1
    rescue LostClaim
      # A replacement owns recovery. This worker must not write another state.
      nil
    rescue StandardError
      terminal = attempts >= MAX_ATTEMPTS
      delay = (2**attempts) + Random.rand(0.0..1.0)
      owned.update_all(state: terminal ? "dead_lettered" : "retrying",
        available_at: Time.current + delay, error_code: "task_failed",
        claim_token: nil, lease_until: nil, updated_at: Time.current)
      Rails.logger.error("ai_task_dead_lettered id=#{id}") if terminal
    end
  end
end
