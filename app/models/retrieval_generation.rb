class RetrievalGeneration < ApplicationRecord
  has_many :document_chunks, dependent: :delete_all
  validates :tenant_id, :dimensions, numericality: { only_integer: true, greater_than: 0 }
  validates :model, :revision, presence: true
  validates :status, inclusion: { in: %w[building ready] }
  validates :vector_column, inclusion: { in: %w[embedding embedding_v2] }
  validate do
    errors.add(:active, 'requires ready generation') if active && status != 'ready'
  end
end
