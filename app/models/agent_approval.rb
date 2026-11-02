class AgentApproval < ApplicationRecord
  SAFE_ACTION_PURPOSE = 'anchor-safe-action-v1'.freeze

  def self.safe_receipt(payload, verifier: Rails.application.message_verifier(SAFE_ACTION_PURPOSE))
    'safe:' + verifier.generate(payload, purpose: SAFE_ACTION_PURPOSE)
  end

  def self.safe_claims(receipt, verifier: Rails.application.message_verifier(SAFE_ACTION_PURPOSE))
    return unless receipt.is_a?(String) && receipt.start_with?('safe:')
    verifier.verified(receipt.delete_prefix('safe:'), purpose: SAFE_ACTION_PURPOSE)
  end

  belongs_to :agent_checkpoint
  validates :tenant_id, :actor_id, :action_key, :nonce, presence: true
  validates :agent_checkpoint_id, :nonce, uniqueness: true
  validates :status, inclusion: { in: %w[approved consumed] }
end
