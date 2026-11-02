require 'json'
require 'time'

module GoldenDatasets
  class Builder
    def initialize(tenant:, diagnostic: ->(record) { AiTelemetry.golden_event(**record) })
      raise ArgumentError, 'explicit_tenant_required' unless tenant.is_a?(String) && GoldenDataset::TOKEN.match?(tenant)
      @tenant, @diagnostic = tenant, diagnostic
    end

    def call(path)
      added = 0
      GoldenDataset.transaction do
        File.foreach(path) do |line|
          raw = JSON.parse(line)
          raise ArgumentError, 'wrong_tenant' unless raw['tenant_id'] == @tenant
          context = raw['context']
          unless context.is_a?(Array) && context.all? { |doc| doc.is_a?(Hash) && doc['tenant_id'] == @tenant }
            raise ArgumentError, 'wrong_tenant_context'
          end
          safe = GoldenDataset.sanitize(raw)
          key = safe.fetch('source_run_id')
          raise ArgumentError, 'invalid_run_id' unless GoldenDataset::TOKEN.match?(key)
          hash = Digest::SHA256.hexdigest(GoldenDataset.canonical(safe))
          previous = GoldenDataset.examples.find_by(tenant_id: @tenant, key: key)
          if previous
            raise ArgumentError, 'source_changed_requires_review' unless previous.payload['intake_sha256'] == hash
            next
          end
          payload = safe.except('tenant_id', 'labeler', 'labeled_at').merge(
            'status' => 'pending', 'review_status' => 'pending', 'needs_review' => false,
            'redacted' => true, 'intake_sha256' => hash)
          GoldenDataset.create!(tenant_id: @tenant, kind: 'example', key: key, payload: payload)
          added += 1
        end
      end
      @diagnostic.call(status: 'ok', count: added)
      added
    rescue StandardError
      @diagnostic.call(status: 'rejected', count: 0)
      raise ArgumentError, 'golden_import_rejected'
    end

    def approve!(keys:, reviewer:)
      raise ArgumentError, 'reviewer_required' unless reviewer.is_a?(String) && GoldenDataset::TOKEN.match?(reviewer)
      GoldenDataset.transaction do
        keys.each do |key|
          row = GoldenDataset.examples.find_by!(tenant_id: @tenant, key: key)
          raise ArgumentError, 'pending_review_required' unless row.payload['status'] == 'pending'
          row.update!(payload: row.payload.merge('status' => 'active', 'review_status' => 'approved',
            'labeler' => reviewer, 'labeled_at' => Time.now.utc.iso8601))
        end
      end
    end

    def relabel!(key:, label:, docs:, reviewer:)
      raise ArgumentError, 'reviewer_required' unless GoldenDataset::TOKEN.match?(reviewer.to_s)
      GoldenDataset.evidence!(@tenant, docs)
      row = GoldenDataset.examples.find_by!(tenant_id: @tenant, key: key)
      ids = row.payload.fetch('source_doc_ids')
      context = ids.map { |id| docs.fetch(id).merge('id' => id) }
      row.update!(payload: GoldenDataset.sanitize(row.payload.merge('context' => context,
        'doc_version' => ids.to_h { |id| [id, docs.fetch(id).fetch('version')] },
        'expected_output' => label, 'status' => 'active', 'review_status' => 'approved',
        'labeler' => reviewer, 'labeled_at' => Time.now.utc.iso8601, 'needs_review' => false)))
    end

    def deprecate!(key:, reviewer:)
      raise ArgumentError, 'reviewer_required' unless GoldenDataset::TOKEN.match?(reviewer.to_s)
      row = GoldenDataset.examples.find_by!(tenant_id: @tenant, key: key)
      row.update!(payload: row.payload.merge('status' => 'deprecated', 'review_status' => 'deprecated',
        'labeler' => reviewer, 'deprecated_at' => Time.now.utc.iso8601, 'needs_review' => false))
    end
  end
end
