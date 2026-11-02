class ToolAction < ApplicationRecord
  belongs_to :order
  has_many :action_reversal_logs
  validates :organization_id, :actor_id, :idempotency_key, :expires_at, presence: true
  validates :status, inclusion: { in: %w[confirmed executed compensated] }
end
