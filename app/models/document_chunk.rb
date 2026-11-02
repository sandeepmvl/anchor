class DocumentChunk < ApplicationRecord
  belongs_to :document
  belongs_to :parent, class_name: 'DocumentChunk', optional: true
  belongs_to :retrieval_generation, optional: true
  validates :tenant_id, numericality: { only_integer: true, greater_than: 0 }
  validates :chunk_index, numericality: { only_integer: true, greater_than_or_equal_to: 0 }
  validates :chunk_index, uniqueness: { scope: [:document_id, :retrieval_generation_id] }
  validates :text, presence: true
  validate :lineage_and_payload

  private

  def lineage_and_payload
    errors.add(:tenant_id, 'document mismatch') unless document && tenant_id == document.tenant_id
    if is_leaf
      unless parent && !parent.is_leaf && parent.document_id == document_id && parent.tenant_id == tenant_id &&
          parent.retrieval_generation_id == retrieval_generation_id
        errors.add(:parent, 'invalid lineage')
      end
      vector = retrieval_generation&.vector_column == 'embedding_v2' ? embedding_v2 : embedding
      if [vector, embedding_model, embedding_revision, embedding_dimensions, window_text].any?(&:blank?)
        errors.add(:base, 'missing leaf payload')
      end
      if retrieval_generation && [tenant_id, embedding_model, embedding_revision, embedding_dimensions] !=
          [retrieval_generation.tenant_id, retrieval_generation.model, retrieval_generation.revision, retrieval_generation.dimensions]
        errors.add(:base, 'generation mismatch')
      end
    elsif parent_id || embedding || embedding_model || embedding_revision || embedding_dimensions || embedding_v2
      errors.add(:base, 'invalid parent payload')
    end
    if [text, window_text].any? { |value| Document.contains_pii?(value) }
      errors.add(:base, 'unredacted_payload')
    end
  end
end
