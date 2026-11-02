class Document < ApplicationRecord
  has_many :document_chunks, dependent: :destroy
  validates :tenant_id, numericality: { only_integer: true, greater_than: 0 }
  validates :external_id, :source_name, :content_hash, :text, presence: true
  validates :external_id, :content_hash, uniqueness: { scope: :tenant_id }
  validates :status, inclusion: { in: %w[processing ready] }
  validate :redacted_payload

  def self.contains_pii?(text)
    rules = YAML.safe_load_file(Rails.root.join('config/retrieval.yml')).fetch('ingestion').fetch('pii_patterns')
    rules.any? { |rule| Regexp.new(rule.fetch('pattern'), Regexp::IGNORECASE).match?(text.to_s) }
  end

  private

  def redacted_payload
    if [text, source_name, external_id].any? { |value| self.class.contains_pii?(value) }
      errors.add(:base, 'unredacted_payload')
    end
    if persisted? && will_save_change_to_tenant_id? && document_chunks.exists?
      errors.add(:tenant_id, 'cannot change after chunk persistence')
    end
  end
end
