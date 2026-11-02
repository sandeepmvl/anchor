class Conversation < ApplicationRecord
  belongs_to :organization
  belongs_to :ticket
  validate { errors.add(:organization, 'mismatch') if ticket && ticket.organization_id != organization_id }
  before_validation :redact_pii_payload

  private

  def redact_pii_payload
    self.pii_payload = PiiRedactionClient.value(pii_payload) if has_attribute?(:pii_payload)
  end
end
