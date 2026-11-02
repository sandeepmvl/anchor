class AgentApproval < ApplicationRecord
  belongs_to :agent_checkpoint
  validates :tenant_id, :actor_id, :action_key, :nonce, presence: true
  validates :agent_checkpoint_id, :nonce, uniqueness: true
  validates :status, inclusion: { in: %w[approved consumed] }
end
