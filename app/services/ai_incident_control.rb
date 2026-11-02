require 'yaml'
require 'time'
require 'securerandom'

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

  # Detection is recorded once; classification and later changes never reset it.
  def self.declare!(organization_id:, signal:, now: Time.now.utc)
    policy = config.fetch('playbook')
    kind = signal.fetch('kind')
    severity = case kind
    when 'cost_spike'
      ratio = signal.fetch('ratio')
      unless ratio.is_a?(Numeric) && ratio.finite? && ratio >= 0
        raise ArgumentError, 'invalid_cost_ratio'
      end
      ratio >= policy.fetch('severity_1_ratio') ? 1 :
        (ratio >= policy.fetch('severity_2_ratio') ? 2 : 3)
    when 'confirmed_harm', 'widespread_unsupported_answers'
      1
    when 'sustained_quality_failure', 'provider_failure'
      2
    when 'isolated_degradation'
      3
    else
      raise ArgumentError, 'unknown_incident_signal'
    end
    # Preserve only operational fields, never raw prompts or credentials.
    safe = signal.slice('kind', 'ratio', 'baseline', 'used', 'window',
      'window_seconds', 'source')
    event = { 'event' => 'incident_declared', 'incident_id' => SecureRandom.uuid,
      'severity' => severity, 'signal' => safe,
      'alert' => policy.fetch('alert_routes').fetch(severity.to_s),
      'detected_at' => now.utc.iso8601(6),
      'human_banner_deadline' => severity == 1 ?
        (now + policy.fetch('human_banner_seconds')).utc.iso8601(6) : nil }
    serialized(organization_id) do |mode|
      mode.update!(revision: mode.revision + 1, audit: mode.audit + [event])
    end
    event
  end

  def self.incident!(organization_id, incident_id)
    serialized(organization_id) do |mode|
      mode.audit.find { |event| event['event'] == 'incident_declared' &&
        event['incident_id'] == incident_id } ||
        raise(ArgumentError, 'incident_not_found')
    end
  end

  def self.contain_incident!(member:, organization_id:, incident_id:)
    incident!(organization_id, incident_id)
    change!(member: member, organization_id: organization_id, human_only: true,
      reason: "incident:#{incident_id}:contain")
  end

  # This is the documented recovery entrypoint, not an automatic reset.
  # Production evidence must come from independent operational checks.
  def self.recover_incident!(member:, organization_id:, incident_id:, checks:,
      production: Rails.env.production?, evidence: ENV)
    incident!(organization_id, incident_id)
    required = %w[quality_pass budget_pass backlog_clear uncertain_usage_reconciled]
    unless checks.is_a?(Hash) && required.all? { |key| checks[key] == true } &&
        checks['observation_seconds'].is_a?(Integer) &&
        checks['observation_seconds'] >= config.fetch('playbook').fetch('recovery_observation_seconds')
      raise ReleaseBlocked, 'incident_recovery_checks_missing'
    end
    release_check!(production: production, evidence: evidence)
    change!(member: member, organization_id: organization_id, human_only: false,
      reason: "incident:#{incident_id}:recovery_checks_passed")
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
