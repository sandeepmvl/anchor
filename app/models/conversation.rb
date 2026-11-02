class Conversation < ApplicationRecord
  belongs_to :organization
  belongs_to :ticket
  validate { errors.add(:organization, 'mismatch') if ticket && ticket.organization_id != organization_id }
end
