class Order < ApplicationRecord
  SAFE_STATE_FIELDS = %w[refunded_cents refund_key status lock_version].freeze

  def refundable_balance_cents
    refund_limit_cents - refunded_cents
  end

  def safe_action_state
    attributes.slice(*SAFE_STATE_FIELDS)
  end

  belongs_to :organization
  validates :reference, format: { with: /\A[A-Z][0-9]{3}\z/ }, uniqueness: { scope: :organization_id }
  validates :payment_id, presence: true
  validates :refund_limit_cents, numericality: { only_integer: true, greater_than: 0 }
  validates :refunded_cents, numericality: { only_integer: true, greater_than_or_equal_to: 0 }
end
