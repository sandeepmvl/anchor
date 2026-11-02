class Ticket < ApplicationRecord
  belongs_to :organization
  belongs_to :user
  has_one :conversation
  has_many :ticket_attachments
  validates :subject, presence: true
end
