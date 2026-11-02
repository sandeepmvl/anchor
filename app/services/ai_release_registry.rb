require 'digest'
require 'json'
require 'open3'

class AiReleaseRegistry
  ROOT = File.expand_path('../..', __dir__)
  FIELDS = %w[contract index model offline_only prompt reply ticket].freeze

  def self.canonical(bundle)
    raise ArgumentError, 'invalid_bundle_fields' unless bundle.keys.sort == FIELDS
    unless bundle['offline_only'] == true && (FIELDS - ['offline_only']).all? { |k| bundle[k].is_a?(String) && !bundle[k].empty? }
      raise ArgumentError, 'offline_fixture_bundle_required'
    end
    FIELDS.to_h { |key| [key, bundle.fetch(key)] }
  end

  def self.identity(bundle)
    Digest::SHA256.hexdigest(JSON.generate(canonical(bundle)))
  end

  def self.create!(tenant_id:, bundle:, contract:)
    id = identity(bundle)
    raise ArgumentError, 'incompatible_bundle' unless bundle['contract'] == contract
    AiReleasePointer.create!(tenant_id: tenant_id, stable: id,
      bundles: { id => canonical(bundle) }, required_contract: contract)
  end

  def initialize(tenant_id:, channel: 'reply', offline: false)
    raise ArgumentError, 'invalid_tenant' unless tenant_id.is_a?(Integer) && tenant_id.positive?
    @tenant_id, @channel, @offline = tenant_id, channel, offline
  end

  def pointer
    AiReleasePointer.find_by!(tenant_id: @tenant_id, channel: @channel)
  end

  def compatible!(row, id)
    bundle = row.bundles.fetch(id)
    raise ArgumentError, 'bundle_identity_changed' unless self.class.identity(bundle) == id
    raise ArgumentError, 'incompatible_bundle_roll_forward_required' unless bundle['contract'] == row.required_contract
    # No trusted live-readiness importer exists in this lesson.
    raise ArgumentError, 'production_readiness_evidence_required' unless @offline && bundle['offline_only'] == true
    bundle
  end

  # The registry revision is persisted with the release pointer, not in process ENV.
  def registry_revision(row)
    event = row.audit.reverse.find { |entry| entry['registry_bundle'] }
    event ? event.fetch('registry_bundle') : row.stable
  end

  def version_consistency!
    row = pointer
    compatible!(row, row.stable)
    raise ArgumentError, 'prompt_version_registry_diverged' unless registry_revision(row) == row.stable
    { 'bundle_id' => row.stable, 'generation' => row.generation,
      'registry_revision' => registry_revision(row), 'consistent' => true }
  end

  def resync(expected_generation:, actor:)
    mutate(expected_generation: expected_generation, actor: actor, action: 'resync') do |row|
      compatible!(row, row.stable)
    end
    version_consistency!
  end

  def mutate(expected_generation:, actor:, action:)
    raise ArgumentError, 'audit_actor_required' if actor.to_s.strip.empty?
    row = pointer
    row.with_lock do
      raise ArgumentError, 'stale_generation' unless row.generation == expected_generation
      before = row.attributes.slice('stable', 'candidate', 'canary_percent')
      yield row
      # The serving registry pointer participates in the same locked transaction.
      row.audit = row.audit + [{ 'registry_bundle' => row.stable,
        'actor' => actor, 'action' => 'registry_sync' }]
      row.generation += 1
      row.audit = row.audit + [{ 'actor' => actor, 'action' => action,
        'at' => Time.now.utc.iso8601, 'generation' => row.generation,
        'before' => before,
        'after' => row.attributes.slice('stable', 'candidate', 'canary_percent') }]
      row.save!
    end
    row.reload
  end

  def stage(bundle:, percent:, expected_generation:, actor:)
    raise ArgumentError, 'invalid_canary_slice' unless percent.is_a?(Integer) && (1..50).cover?(percent)
    id = self.class.identity(bundle)
    mutate(expected_generation: expected_generation, actor: actor, action: 'stage') do |row|
      row.bundles = row.bundles.merge(id => self.class.canonical(bundle))
      compatible!(row, id)
      raise ArgumentError, 'candidate_equals_stable' if id == row.stable
      row.candidate, row.canary_percent = id, percent
    end
  end

  def pin(conversation_id:)
    raise ArgumentError, 'conversation_required' if conversation_id.to_s.empty?
    row = pointer # One SELECT reads the entire routing snapshot, including bundles.
    bucket = Digest::SHA256.hexdigest("#{@tenant_id}:#{conversation_id}").to_i(16) % 100
    id = row.candidate && bucket < row.canary_percent ? row.candidate : registry_revision(row)
    bundle = compatible!(row, id)
    JSON.parse(JSON.generate({ 'bundle_id' => id, 'generation' => row.generation,
      'tenant_id' => @tenant_id, 'bundle' => bundle,
      'active_bundle_id' => row.stable, 'registry_revision' => registry_revision(row),
      'cache_namespace' => id })).freeze
  end

  def with_request(conversation_id:)
    pinned = pin(conversation_id: conversation_id)
    PromptRegistry.with_release(pinned) { yield pinned }
  end

  def health!(row, health)
    raise ArgumentError, 'health_identity_mismatch' unless health['generation'] == row.generation && health['candidate'] == row.candidate
    %w[control_count candidate_count].each do |key|
      raise ArgumentError, 'insufficient_scored_interactions' unless health[key].is_a?(Integer) && health[key] >= 4
    end
    %w[control_quality candidate_quality error_rate p95_ms window_seconds].each do |key|
      value = health[key]
      raise ArgumentError, 'invalid_health_metric' unless value.is_a?(Numeric) && value.finite?
    end
    unless (0.0..1.0).cover?(health['control_quality']) &&
        (0.0..1.0).cover?(health['candidate_quality']) &&
        (0.0..0.01).cover?(health['error_rate']) &&
        (0..2000).cover?(health['p95_ms']) && health['window_seconds'] >= 300 &&
        health['candidate_quality'] >= 0.8 &&
        health['control_quality'] - health['candidate_quality'] <= 0.1
      raise ArgumentError, 'canary_health_gate_failed'
    end
  end

  def promote(expected_generation:, actor:, health:)
    mutate(expected_generation: expected_generation, actor: actor, action: 'promote') do |row|
      raise ArgumentError, 'candidate_required' unless row.candidate
      bundle = compatible!(row, row.candidate)
      health!(row, health)
      output, status = Open3.capture2('python3', File.join(ROOT, 'ops/deploy/release_bundle.py'),
        '--gate', JSON.generate(bundle), chdir: ROOT)
      raise ArgumentError, 'golden_and_judge_gate_failed' unless status.success? && JSON.parse(output).fetch('passed')
      row.previous, row.stable = row.stable, row.candidate
      row.candidate, row.canary_percent = nil, 0
    end
  end

  def revert(expected_generation:, actor:)
    mutate(expected_generation: expected_generation, actor: actor, action: 'revert') do |row|
      # A rejected canary leaves stable unchanged. A promoted release has a previous target.
      target = row.candidate ? row.stable : row.previous
      raise ArgumentError, 'no_revert_target' unless target
      compatible!(row, target)
      row.previous, row.stable = row.stable, target if target != row.stable
      row.candidate, row.canary_percent = nil, 0
    end
  end
end
