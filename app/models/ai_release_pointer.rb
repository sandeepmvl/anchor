class AiReleasePointer < ApplicationRecord
  validates :tenant_id, numericality: { only_integer: true, greater_than: 0 }
  validates :channel, :stable, :required_contract, presence: true
  validates :channel, uniqueness: { scope: :tenant_id }
  validates :canary_percent, inclusion: { in: 0..100 }
end
