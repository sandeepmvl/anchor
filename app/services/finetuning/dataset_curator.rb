require 'json'
require 'yaml'
require 'date'
require 'time'
require 'digest'
require 'open3'
require 'fileutils'

module Finetuning
  class DatasetCurator
    ROOT = File.expand_path('../../..', __dir__)

    def initialize(tenant:, since:, as_of:, source:, out:, python: ENV.fetch('ANCHOR_PYTHON', 'python3'))
      raise ArgumentError, 'explicit_tenant_required' unless tenant.is_a?(String) && tenant.match?(/\A[a-zA-Z0-9_-]+\z/)
      raise ArgumentError, 'invalid_window' unless since.match?(/\A[1-9][0-9]*d\z/)
      @tenant, @source, @out, @python = tenant, source, out, python
      @end = Time.utc(*Date.iso8601(as_of).to_s.split('-').map(&:to_i)) + 86400
      @start = @end - 86400 - since.to_i * 86400
      @policy = YAML.load_file(File.join(ROOT, 'config/fine_tuning.yml')).fetch('dataset')
    end

    def normalize(text)
      text.downcase.scan(/[[:alnum:]_]+/).join(' ')
    end

    def near?(left, right)
      a, b = [left, right].map { |s| normalize(s).split.uniq }
      return false unless left.scan(/\d+/) == right.scan(/\d+/)
      return false unless (a & %w[not no never]) == (b & %w[not no never])
      union = (a | b).length
      union.positive? && (a & b).length.to_f / union >= @policy.fetch('near_duplicate_threshold')
    end

    def call
      review_path = @source.sub(/\.jsonl\z/, '_manifest.json')
      review = JSON.parse(File.read(review_path))
      unless review['tenant_id'] == @tenant && review['training_permitted'] == true &&
             review['rubric_version'] == @policy['rubric_version'] && review['reviewer'].is_a?(String)
        raise ArgumentError, 'unauthorized_source_manifest'
      end
      counts = Hash.new(0)
      eligible = []
      audit = []
      seen_ids = {}
      File.foreach(@source) do |line|
        row = JSON.parse(line)
        if row.key?('expected_output') || row.key?('review_status') || row['evaluation_only'] == true
          raise ArgumentError, 'golden_evaluation_data_forbidden_in_training'
        end
        if row['tenant_id'] != @tenant
          counts['wrong_tenant'] += 1
          next
        end
        %w[id conversation_id family_id].each do |field|
          raise ArgumentError, 'invalid_opaque_id' unless row[field].is_a?(String) && row[field].match?(/\A[a-zA-Z0-9_-]+\z/)
        end
        raise ArgumentError, 'repeated_source_id' if seen_ids[row['id']]
        seen_ids[row['id']] = true
        at = Time.iso8601(row.fetch('at'))
        reason = if at < @start || at >= @end
          'outside_window'
        elsif !review.fetch('approved_ids').include?(row['id'])
          'review_deferred'
        elsif !review.fetch('pii_reviewed_ids').include?(row['id'])
          'pii_deferred'
        elsif !@policy['labels'].include?(row['label'])
          'label_deferred'
        elsif !%w[input output].all? { |k| row[k].is_a?(String) && @policy["#{k}_length"].then { |lo, hi| row[k].length.between?(lo, hi) } }
          'length'
        elsif !row['input'].include?('Policy: returns within 30 days.') && !row['input'].include?('POLICY: RETURNS WITHIN 30 DAYS.')
          'context_deferred'
        end
        if reason
          counts[reason] += 1
          audit << {'id' => row['id'], 'reason' => reason}
        else
          eligible << row
        end
      end
      eligible.sort_by! { |r| r['id'] }
      payload = eligible.map { |r| r.slice('input', 'output', 'label') }
      stdout, _stderr, status = Open3.capture3(@python, File.join(ROOT, 'bin/redact_finetuning.py'), stdin_data: JSON.generate(payload))
      raise 'finetuning_redaction_failed' unless status.success?
      safe = JSON.parse(stdout)
      raise 'invalid_redaction_batch' unless safe.is_a?(Array) && safe.length == eligible.length
      safe.each { |r| raise 'invalid_redaction_schema' unless r.keys.sort == %w[input label output] }
      # Raw transcript text is no longer used after the boundary.
      metadata = eligible.map { |r| r.slice('id', 'conversation_id', 'family_id', 'at') }
      eligible = payload = nil
      parent = (0...safe.length).to_a
      find = lambda { |i| parent[i] == i ? i : parent[i] = find.call(parent[i]) }
      join = lambda { |i, j| parent[find.call(j)] = find.call(i) }
      dropped = {}
      conflict = false
      safe.each_index do |i|
        (0...i).each do |j|
          same_group = %w[conversation_id family_id].any? { |key| metadata[i][key] == metadata[j][key] }
          exact = safe[i] == safe[j]
          normalized = %w[input output label].all? { |k| normalize(safe[i][k]) == normalize(safe[j][k]) }
          similar = near?(safe[i]['input'], safe[j]['input'])
          join.call(j, i) if same_group || exact || normalized || similar
          next unless exact || normalized || similar
          compatible = safe[i]['label'] == safe[j]['label'] && normalize(safe[i]['output']) == normalize(safe[j]['output'])
          conflict = true unless compatible
          next if dropped[i] || dropped[j]
          reason = exact ? 'exact_duplicate' : normalized ? 'normalized_duplicate' : 'near_duplicate'
          dropped[i] = {'reason' => reason, 'matched_id' => metadata[j]['id']}
        end
      end
      raise 'conflicting_targets_require_review' if conflict
      groups = safe.each_index.group_by { |i| find.call(i) }.values.sort_by { |indices| indices.map { |i| metadata[i]['id'] }.min }
      assignments = {}
      groups.each_with_index do |indices, index|
        split = @policy['splits'][index % @policy['splits'].length]
        group_id = "group-#{index + 1}"
        indices.each { |i| assignments[i] = {'split' => split, 'group_id' => group_id} }
      end
      splits = @policy['splits'].to_h { |split| [split, []] }
      safe.each_index do |i|
        info = assignments.fetch(i)
        rejection = dropped[i]
        reason = rejection ? rejection['reason'] : 'accepted'
        counts[reason] += 1
        entry = metadata[i].merge(info).merge('reason' => reason)
        entry['matched_id'] = rejection['matched_id'] if rejection
        unless rejection
          entry['row_index'] = splits[info['split']].length
          splits[info['split']] << safe[i]
        end
        audit << entry
      end
      files = @policy['splits'].to_h do |split|
        path = split == 'train' ? @out : File.join(File.dirname(@out), "#{split}.jsonl")
        [split, path]
      end
      manifest = {
        'tenant_id' => @tenant, 'rubric_version' => @policy['rubric_version'],
        'as_of' => (@end - 86400).utc.iso8601, 'status' => @policy['status'],
        'source_sha256' => Digest::SHA256.file(@source).hexdigest,
        'review_manifest_sha256' => Digest::SHA256.file(review_path).hexdigest,
        'scope' => review['scope'], 'counts' => counts, 'audit' => audit,
        'split_strategy' => 'sorted_connected_groups_round_robin_v1',
        'label_histograms' => splits.transform_values { |rows| rows.group_by { |r| r['label'] }.transform_values(&:length) },
        'files' => files, 'export_sha256' => {}
      }
      # Only redacted rows reach storage. Publish the manifest last.
      FileUtils.mkdir_p(File.dirname(@out))
      files.each do |split, path|
        content = splits[split].map { |row| JSON.generate(row) + "\n" }.join
        File.write(path, content)
        manifest['export_sha256'][split] = Digest::SHA256.hexdigest(content)
      end
      File.write(@out.sub(/\.jsonl\z/, '') + '.manifest.json', JSON.pretty_generate(manifest))
      manifest
    end
  end
end
