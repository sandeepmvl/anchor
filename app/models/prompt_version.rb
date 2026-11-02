class PromptVersion < ApplicationRecord
  validates :version, format: { with: /\Achat:v[1-9][0-9]*\z/ }, uniqueness: true
  validates :template, :model_id, presence: true
  validates :model_id, format: { with: /\A[a-zA-Z0-9.-]+-\d{4}-\d{2}-\d{2}\z/ }
  validates :temperature, numericality: { greater_than_or_equal_to: 0, less_than_or_equal_to: 2 }
  validates :max_tokens, numericality: { only_integer: true, greater_than: 0 }
  validate do
    errors.add(:template, 'must contain one ticket placeholder') unless template.to_s.scan('{{ticket}}').length == 1
  end

  def readonly?
    persisted? || super
  end
end
