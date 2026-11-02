require 'yaml'
require 'digest'

class AnswerQualitySample < ApplicationRecord
  def self.policy
    YAML.safe_load(File.read(File.expand_path('../../config/online_quality.yml', __dir__)))
  end

  # Call only with identity resolved by the authenticated completion boundary.
  def self.capture!(organization:, message_id:, answer_version:, intent:,
                    model_version:, trace_id:, event_at:, payload:)
    p = policy
    rate = Float(p.fetch('sample_rate'))
    limit = Integer(p.fetch('daily_tenant_limit'))
    raise ArgumentError, 'invalid quality budget' unless rate.between?(0, 1) && limit.positive?
    key = {organization_id: organization.id, message_id: message_id.to_s,
           answer_version: answer_version.to_s, evaluator_version: p.fetch('evaluator_version')}
    identity = [organization.id, message_id, answer_version, p.fetch('evaluator_version')].join(':')
    fraction = Digest::SHA256.hexdigest(p.fetch('seed') + ':' + identity)[0, 13].to_i(16).fdiv(16**13)
    return unless fraction < rate
    organization.with_lock do
      existing = find_by(key)
      return existing if existing
      day = Time.now.utc.to_date
      return if where(organization_id: organization.id, budget_day: day).count >= limit
      safe = PiiRedactionClient.value(payload)
      create!(**key, intent: intent, model_version: model_version, trace_id: trace_id,
              event_at: event_at, budget_day: day, redacted_payload: safe)
    end
  end

  def self.attach_feedback!(feedback:, message_id:)
    # GoldenCandidate feedback stores its authorized identity as tenant_id.
    # Also accept organization_id for callers using the Rails organization key.
    # Never use a tenant identifier supplied separately by the request.
    identities = feedback.attributes.values_at('tenant_id', 'organization_id')
      .compact.map(&:to_s).reject(&:empty?).uniq
    raise ActiveRecord::RecordNotFound unless identities.length == 1
    tenant = identities.first
    where(organization_id: tenant, message_id: message_id.to_s)
      .update_all(feedback_event_id: feedback.id, feedback_at: Time.now.utc)
  end

  def self.enqueue_pending!
    where(status: 'pending').find_each do |sample|
      ScoreAnswerQualityJob.perform_later(sample.organization_id, sample.id)
    end
  end
end
