class AiBudgetReservation < ApplicationRecord
  validates :attempt_id, :actor, :feature, :day, presence: true
  validates :tenant_id, :bound, numericality: { only_integer: true, greater_than: 0 }
  validates :state, inclusion: { in: %w[pending settled] }
  # Mutation belongs to AiQuotaPolicy's serialized authority transaction.
end
