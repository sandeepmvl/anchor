require 'time'

module GoldenDatasets
  class Versioner
    def initialize(tenant:)
      raise ArgumentError, 'explicit_tenant_required' unless GoldenDataset::TOKEN.match?(tenant.to_s)
      @tenant = tenant
    end

    def revalidate!(docs:)
      GoldenDataset.evidence!(@tenant, docs)
      flagged = []
      examples.each do |row|
        next unless row.payload['status'] == 'active'
        p = row.payload
        stale = p.fetch('context').any? do |old|
          current = docs[old.fetch('id')]
          !current || current['version'] != old['version'] || current['text'] != old['text']
        end
        flagged << row.key if stale
        row.update!(payload: p.merge('needs_review' => stale)) if p['needs_review'] != stale
      end
      flagged.sort
    end

    def publish!(name:, against:, docs:, built_against_tag:)
      raise ArgumentError, 'invalid_version_name' unless GoldenDataset::TOKEN.match?(name.to_s)
      raise ArgumentError, 'release_tag_required' unless GoldenDataset::TOKEN.match?(built_against_tag.to_s)
      # Persist the review flags even when publication is refused.
      raise ArgumentError, 'stale_examples_require_review' unless revalidate!(docs: docs).empty?
      GoldenDataset.transaction do
        rows = examples.select { |r| r.payload['status'] == 'active' }.map do |row|
          row.validate!
          raise ArgumentError, 'approval_required' unless row.payload['review_status'] == 'approved'
          {'id' => row.key}.merge(row.payload)
        end
        rows.sort_by! { |row| row.fetch('id') }
        raise ArgumentError, 'empty_dataset' if rows.empty?
        old = against ? GoldenDataset.versions.find_by!(tenant_id: @tenant, key: against).payload.fetch('rows') : []
        before = old.to_h { |row| [row.fetch('id'), row] }
        after = rows.to_h { |row| [row.fetch('id'), row] }
        removed = (before.keys - after.keys).sort
        removed.each do |key|
          row = examples.find { |item| item.key == key }
          raise ArgumentError, 'deprecation_history_required' unless row && row.payload['status'] == 'deprecated'
        end
        diff = {'added' => (after.keys - before.keys).sort,
          'changed' => (before.keys & after.keys).select { |id| before[id] != after[id] }.sort,
          'removed' => removed}
        data = {'rows' => rows, 'sha256' => Digest::SHA256.hexdigest(GoldenDataset.canonical(rows)),
          'against' => against, 'diff' => diff, 'built_at' => Time.now.utc.iso8601,
          'built_against_tag' => built_against_tag}
        existing = GoldenDataset.versions.find_by(tenant_id: @tenant, key: name)
        if existing
          unless %w[rows sha256 against diff built_against_tag].all? { |k| existing.payload[k] == data[k] }
            raise ArgumentError, 'published_name_conflict'
          end
          return existing
        end
        GoldenDataset.create!(tenant_id: @tenant, kind: 'version', key: name, payload: data)
      end
    end

    def coverage
      examples.group_by { |row| row.payload['status'] }.transform_values do |group|
        payloads = group.map(&:payload)
        {'count' => group.length,
          'tags' => payloads.flat_map { |p| p.fetch('tags') }.tally,
          'difficulty' => payloads.map { |p| p.fetch('difficulty') }.tally}
      end
    end

    def self.replay_evidence(path, tenant:)
      docs = {}
      File.foreach(path) do |line|
        row = JSON.parse(line)
        raise ArgumentError, 'wrong_tenant' unless row['tenant_id'] == tenant
        row.fetch('context').each do |doc|
          id = doc.fetch('id')
          value = doc.except('id')
          raise ArgumentError, 'conflicting_replay_evidence' if docs.key?(id) && docs[id] != value
          docs[id] = value
        end
      end
      GoldenDataset.evidence!(tenant, docs)
    end

    private

    def examples
      GoldenDataset.examples.where(tenant_id: @tenant).order(:key).to_a
    end
  end
end
