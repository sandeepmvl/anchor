class Membership < ApplicationRecord
  belongs_to :organization
  belongs_to :user
  validates :role, inclusion: { in: %w[admin agent end_customer] }
  validates :user_id, uniqueness: { scope: :organization_id }
end
