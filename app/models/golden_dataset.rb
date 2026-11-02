require 'json'
require 'digest'

class GoldenDataset < ActiveRecord::Base
  self.table_name = 'golden_datasets'
  TOKEN = /\A[a-zA-Z0-9_-]{1,80}\z/
  REQUIRED = %w[query context expected_output tags difficulty source case_type
    source_run_id provenance source_doc_ids doc_version status review_status redacted needs_review].freeze
  scope :examples, -> { where(kind: 'example') }
  scope :versions, -> { where(kind: 'version') }
  validates :tenant_id, :key, format: { with: TOKEN }
  validates :kind, inclusion: { in: %w[example version] }
  validates :key, uniqueness: { scope: [:tenant_id, :kind] }
  validate :contract
  before_update :protect_version
  before_destroy :protect_version

  def self.canonical(value)
    case value
    when Hash then '{' + value.keys.sort.map { |k| JSON.generate(k) + ':' + canonical(value[k]) }.join(',') + '}'
    when Array then '[' + value.map { |v| canonical(v) }.join(',') + ']'
    else JSON.generate(value)
    end
  end

  # These fields contain fingerprints or audit times, not customer content.
  SANITIZER_METADATA = %w[intake_sha256 sha256 labeled_at built_at deprecated_at].freeze

  def self.sanitize(value)
    case value
    when Hash
      value.to_h do |key, item|
        [key, SANITIZER_METADATA.include?(key) ? item : sanitize(item)]
      end
    when Array then value.map { |v| sanitize(v) }
    when String
      value.gsub(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i, '[EMAIL]')
        # Remove a complete formatted phone before a card regex can consume
        # the phone's final group together with the following card's groups.
        .gsub(/(?<!\d)(?:\+?1[ .()-]+)?(?:\(\d{3}\)|\d{3})[ .-]+\d{3}[ .-]+\d{4}(?!\d)/, '[PHONE]')
        .gsub(/\b(?:\d{4}[ -]){3}\d{4}\b|\b\d{13,19}\b/, '[CARD]')
        .gsub(/(?<!\d)(?:\+?\d[ ().-]?){9,14}\d(?!\d)/, '[PHONE]')
    else value
    end
  end

  def self.evidence!(tenant, docs)
    raise ArgumentError, 'current_evidence_required' unless docs.is_a?(Hash) && !docs.empty?
    docs.each do |id, doc|
      unless id.match?(TOKEN) && doc.is_a?(Hash) && doc['tenant_id'] == tenant &&
          doc['version'].is_a?(Integer) && doc['version'].positive? && doc['text'].is_a?(String)
        raise ArgumentError, 'unauthorized_evidence'
      end
    end
    docs
  end

  def contract
    p = payload
    unless p.is_a?(Hash) && self.class.sanitize(p) == p
      errors.add(:payload, 'must be sanitized'); return
    end
    if kind == 'version'
      errors.add(:payload, 'invalid snapshot') unless p['rows'].is_a?(Array) &&
        p['sha256'] == Digest::SHA256.hexdigest(self.class.canonical(p['rows']))
      return
    end
    errors.add(:payload, 'missing fields') unless REQUIRED.all? { |k| p.key?(k) }
    %w[query expected_output source source_run_id].each do |field|
      errors.add(:payload, "missing #{field}") unless p[field].is_a?(String) && !p[field].strip.empty?
    end
    errors.add(:payload, 'invalid run ID') unless TOKEN.match?(p['source_run_id'].to_s)
    errors.add(:payload, 'invalid tags') unless p['tags'].is_a?(Array) &&
      !p['tags'].empty? && p['tags'].all? { |tag| tag.is_a?(String) && TOKEN.match?(tag) }
    errors.add(:payload, 'invalid difficulty') unless %w[easy medium hard].include?(p['difficulty'])
    errors.add(:payload, 'invalid case type') unless p['case_type'] == 'rag' && p['expected_tool_calls'].nil?
    states = {'pending' => 'pending', 'active' => 'approved', 'deprecated' => 'deprecated'}
    errors.add(:payload, 'invalid review state') unless states.key?(p['status']) && states[p['status']] == p['review_status']
    errors.add(:payload, 'redaction required') unless p['redacted'] == true
    errors.add(:payload, 'invalid staleness flag') unless [true, false].include?(p['needs_review'])
    provenance = p['provenance']
    errors.add(:payload, 'provenance required') unless provenance.is_a?(Hash) &&
      %w[replay reviewed_seed production].include?(provenance['kind']) && TOKEN.match?(provenance['reference'].to_s)
    ids, versions, context = p.values_at('source_doc_ids', 'doc_version', 'context')
    valid = ids.is_a?(Array) && !ids.empty? && ids.uniq == ids && versions.is_a?(Hash) &&
      versions.keys.sort == ids.sort && context.is_a?(Array) && context.length == ids.length
    valid &&= context.all? do |doc|
      doc.is_a?(Hash) && ids.include?(doc['id']) && doc['tenant_id'] == tenant_id &&
        doc['text'].is_a?(String) && !doc['text'].empty? &&
        doc['version'].is_a?(Integer) && doc['version'].positive? && versions[doc['id']] == doc['version']
    end
    valid &&= context.map { |doc| doc['id'] }.sort == ids.sort
    errors.add(:payload, 'invalid tenant evidence') unless valid
    if p['status'] == 'active'
      errors.add(:payload, 'review identity required') unless TOKEN.match?(p['labeler'].to_s) && p['labeled_at'].is_a?(String)
    end
  end

  def protect_version
    raise ActiveRecord::ReadOnlyRecord, 'immutable golden version' if kind_in_database == 'version'
  end
end
