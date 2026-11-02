class Organization < ApplicationRecord
  has_many :memberships
  has_many :tickets
  has_many :conversations
  has_many :ticket_attachments
  has_many :orders
  validates :name, presence: true
end
