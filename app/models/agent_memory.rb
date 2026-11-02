class AgentMemory < ApplicationRecord
  EMAIL = /[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}/i
  PHONE = /(?<!\w)\+?\d[\d ().\-]{7,}\d(?!\w)/

  def self.redact(text)
    text.to_s.gsub(EMAIL, '[EMAIL]').gsub(PHONE, '[PHONE]')
  end

  before_validation { self.summary = self.class.redact(summary) }
  validates :tenant_id, :conversation_id,
    numericality: { only_integer: true, greater_than: 0 }
  validates :run_key, presence: true, length: { maximum: 100 },
    format: { with: /\A[a-zA-Z0-9_-]+\z/ }
  validates :summary, presence: true, length: { maximum: 16000 }

  def public_record
    { conversation_id: conversation_id, run_key: run_key, summary: summary }
  end
end
