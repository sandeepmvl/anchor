class AgentCheckpoint < ApplicationRecord
  has_one :agent_approval, dependent: :restrict_with_exception
  validates :tenant_id, :conversation_id, :thread_id, :action_key, :context_json, presence: true
  validates :thread_id, uniqueness: true
  validates :status, inclusion: { in: %w[pending approved completed] }
end
