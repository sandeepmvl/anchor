class TicketAttachment < ApplicationRecord
  belongs_to :organization
  belongs_to :ticket
  has_one_attached :evidence_file
  validates :kind, inclusion: { in: %w[screenshot scanned_page voice] }
  validates :status, inclusion: { in: %w[queued answered unreadable error] }
  validate :consistent_scope

  private

  def consistent_scope
    errors.add(:organization, 'mismatch') if ticket && ticket.organization_id != organization_id
    errors.add(:evidence, 'unexpected') if %w[unreadable error].include?(status) && evidence.present?
  end
end
