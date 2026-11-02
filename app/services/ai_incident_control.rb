require 'yaml'

class AiIncidentControl
  class Unauthorized < StandardError; end
  class HumanOnly < StandardError; end
  class ReleaseBlocked < StandardError; end

  def self.config
    YAML.safe_load(File.read(File.expand_path('../../config/incident_control.yml', __dir__)))
  end

  def self.release_check!(production: Rails.env.production?, evidence: ENV)
    return unless production
    unless config.fetch('production_evidence').all? { |key| evidence[key] == '1' }
      raise ReleaseBlocked, 'incident_release_evidence_missing'
    end
  end

  # Always take this lock before order/action locks. Do not cache the mode.
  # Holding it across dispatch gives a precise local acknowledgement boundary.
  # Production must validate this ordering with its actual database and adapters.
  def self.serialized(organization_id)
    Organization.transaction do
      organization = Organization.lock.find(organization_id)
      mode = AiOperatingMode.find_or_create_by!(organization_id: organization.id)
      mode.lock!
      yield mode
    end
  end

  def self.change!(member:, organization_id:, human_only:, reason:)
    unless member.is_a?(Membership) && member.persisted? &&
        [true, false].include?(human_only) && reason.is_a?(String) &&
        reason.strip.length.between?(1, 500)
      raise Unauthorized, 'incident_change_denied'
    end
    serialized(organization_id) do |mode|
      member.reload
      unless member.organization_id == organization_id && member.role == 'admin'
        raise Unauthorized, 'incident_change_denied'
      end
      mode.update!(human_only: human_only, revision: mode.revision + 1,
        audit: mode.audit + [{ 'actor_id' => member.user_id,
          'human_only' => human_only, 'reason' => reason.strip,
          'at' => Time.now.utc.iso8601 }])
      mode
    end
  end

  def self.human_only?(organization_id)
    serialized(organization_id) { |mode| mode.human_only }
  end

  def self.with_ai_permission!(organization_id)
    serialized(organization_id) do |mode|
      raise HumanOnly, 'human_only' if mode.human_only
      yield
    end
  end

  def self.handoff(context, reason: 'human_only')
    # Callers supply already-sanitized context from durable product storage.
    # Never transfer the signed execution_context, headers, or credentials.
    allowed = %w[message history evidence proposed_action order_id amount_cents
                 run_id ticket_id conversation_id attempts idempotency_key]
    safe = context.to_h.transform_keys(&:to_s).slice(*allowed)
    { 'status' => 'human_only', 'handoff' => true, 'reason' => reason,
      'reply' => config.fetch('message'), 'context' => safe }
  end

  def self.run_agent(organization_id:, payload:, key:)
    with_ai_permission!(organization_id) { yield }
  rescue HumanOnly
    request = payload.fetch('request').to_h
    context = request.merge('idempotency_key' => key)
    handoff(context)
  end
end
