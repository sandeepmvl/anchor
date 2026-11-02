class Order < ApplicationRecord
  belongs_to :organization
  validates :reference, format: { with: /\A[A-Z][0-9]{3}\z/ }, uniqueness: { scope: :organization_id }
  validates :payment_id, presence: true
  validates :refund_limit_cents, numericality: { only_integer: true, greater_than: 0 }
  validates :refunded_cents, numericality: { only_integer: true, greater_than_or_equal_to: 0 }
end
