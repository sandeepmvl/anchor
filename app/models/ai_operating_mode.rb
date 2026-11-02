class AiOperatingMode < ApplicationRecord
  belongs_to :organization
  validates :organization_id, uniqueness: true
  validates :revision, numericality: { only_integer: true, greater_than_or_equal_to: 0 }
end
