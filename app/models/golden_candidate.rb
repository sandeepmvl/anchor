require 'json'

class GoldenCandidate < ApplicationRecord
  class AuthorizationError < StandardError; end
  class InvalidInput < StandardError; end

  class AnswerMessage < ApplicationRecord
    self.table_name = 'answer_messages'
  end

  class FeedbackEvent < ApplicationRecord
    self.table_name = 'feedback_events'
  end

  belongs_to :answer_message, class_name: 'GoldenCandidate::AnswerMessage'
  validates :status, inclusion: {in: %w[queued promoted]}

  def self.verifier
    Rails.application.message_verifier('anchor_answer_context')
  end

  def self.claims!(token, purpose:)
    claims = verifier.verified(token.to_s, purpose: purpose)
    unless claims.is_a?(Hash) && claims['expires_at'].is_a?(Integer) &&
        claims['expires_at'] > Time.now.to_i && claims['expires_at'] <= Time.now.to_i + 3600 &&
        claims['tenant_id'].to_s.match?(/\A[1-9][0-9]*\z/) &&
        claims['subject'].to_s.match?(/\A[a-zA-Z0-9_-]{1,100}\z/)
      raise AuthorizationError, 'verified_context_required'
    end
    if purpose == 'answer_customer' && !claims['conversation_id'].to_s.match?(/\A[a-zA-Z0-9_-]{1,100}\z/)
      raise AuthorizationError, 'conversation_ownership_required'
    end
    if purpose == 'golden_reviewer' && claims['role'] != 'golden_reviewer'
      raise AuthorizationError, 'reviewer_required'
    end
    claims
  end

  def self.redact(value)
    # Fail closed on common direct identifiers; reuse the same policy for all writes.
    case value
    when Hash
      value.to_h.transform_keys(&:to_s).transform_values { |item| redact(item) }
    when Array
      value.map { |item| redact(item) }
    when String
      value.gsub(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i, '[EMAIL]')
        .gsub(/\b(?:\+?\d[\d ().-]{7,}\d)\b/, '[PHONE]')
        .gsub(/\b(?:\d[ -]*?){13,19}\b/, '[PAYMENT]')
    else
      value
    end
  end

  def self.safe!(value)
    safe = redact(value)
    # Unset Rails custom options can answer arbitrary methods via method_missing.
    # Require a real call method, not merely respond_to?(:call).
    policy = Rails.configuration.x.answer_redactor
    callable = policy.class.public_method_defined?(:call) ||
      policy.singleton_class.public_method_defined?(:call)
    if Rails.env.production? && !callable
      raise InvalidInput, 'production_redaction_not_configured'
    end
    if callable
      result = policy.call(safe)
      unless result.is_a?(safe.class)
        raise InvalidInput, 'invalid_redaction_result'
      end
      safe = redact(result)
    end
    safe
  end

  def self.feedback!(claims:, message_id:, comment:)
    raise InvalidInput, 'comment_too_long' if comment.to_s.length > 2000
    message = AnswerMessage.find_by!(id: message_id, tenant_id: claims.fetch('tenant_id').to_s,
      customer_id: claims.fetch('subject'), conversation_id: claims.fetch('conversation_id'))
    feedback = candidate = nil
    message.with_lock do
      safe_comment = safe!(comment.to_s)
      feedback = FeedbackEvent.find_or_initialize_by(answer_message_id: message.id)
      feedback.assign_attributes(tenant_id: message.tenant_id, rating: 'down', comment: safe_comment)
      feedback.save!
      candidate = find_or_initialize_by(answer_message_id: message.id)
      if candidate.new_record?
        candidate.assign_attributes(tenant_id: message.tenant_id,
          conversation_id: message.conversation_id, prompt_version: message.prompt_version,
          retrieved_chunk_ids: message.retrieved_chunk_ids,
          payload: safe!(message.payload.merge('comment' => safe_comment)), status: 'queued')
      elsif candidate.status == 'queued'
        candidate.payload = safe!(candidate.payload.merge('comment' => safe_comment))
      end
      candidate.save!
    end
    [feedback, candidate]
  end
end
