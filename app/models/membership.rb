class Membership < ApplicationRecord
  belongs_to :organization
  belongs_to :user
  validates :role, inclusion: { in: %w[admin agent end_customer] }
  validates :user_id, uniqueness: { scope: :organization_id }

  def tool_action_allowed?(action, grants)
    grants.fetch(role, []).include?(action)
  end
end
