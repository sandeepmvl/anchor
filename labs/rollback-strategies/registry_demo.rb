require_relative 'test_support'
require 'open3'

abort 'usage: --stage | --canary | --revert | --probe' unless ARGV.length == 1 && %w[--stage --canary --revert --probe].include?(ARGV[0])
ReleaseLab.connect(File.join(ReleaseLab::ROOT, '.anchor-release-demo.sqlite3'))
bundles = ReleaseLab.bundles
unless AiReleasePointer.exists?(tenant_id: 1, channel: 'reply')
  AiReleaseRegistry.create!(tenant_id: 1, bundle: bundles.fetch('good'), contract: 'support-tools:v1')
end
registry = ReleaseLab.registry
row = registry.pointer
case ARGV[0]
when '--stage'
  # Publishing uses the existing immutable prompt registry in the Rails app.
  # This standalone fixture answers from the bundle and does not create PromptVersion rows.
  if row.candidate
    abort 'candidate_already_staged; revert before staging again'
  end
  row = registry.stage(bundle: bundles.fetch('degraded'), percent: 20,
    expected_generation: row.generation, actor: 'offline-operator')
  puts JSON.pretty_generate(row.attributes.slice('stable', 'candidate', 'generation', 'canary_percent'))
when '--canary'
  abort 'stage a candidate first' unless row.candidate
  samples = 100.times.map { |n| registry.pin(conversation_id: "ticket-#{n}") }
  candidate_count = samples.count { |p| p['bundle_id'] == row.candidate }
  abort 'fixture population missed a slice' unless candidate_count >= 4 && candidate_count <= 96
  health = ReleaseLab.health(row).merge('candidate_quality' => 0.2)
  begin
    registry.promote(expected_generation: row.generation, actor: 'offline-operator', health: health)
    abort 'bad canary unexpectedly promoted'
  rescue ArgumentError => error
    raise unless error.message == 'canary_health_gate_failed'
    puts "health gate: BLOCK (#{error.message})"
  end
  gate, status = Open3.capture2('python3', 'ops/deploy/release_bundle.py', '--gate', JSON.generate(bundles.fetch('degraded')), chdir: ReleaseLab::ROOT)
  abort 'gate process failed' unless status.success?
  verdict = JSON.parse(gate)
  abort 'golden gate unexpectedly passed' if verdict.fetch('passed')
  puts "golden/regression gate: BLOCK #{verdict.fetch('errors').join(',')}"
  abort 'stable changed' unless registry.pointer.stable == row.stable
  puts JSON.generate(control_count: 100 - candidate_count, candidate_count: candidate_count,
    stable_unchanged: true, quality_basis: 'constructed_fixture_not_live_scores')
when '--revert'
  row = registry.revert(expected_generation: row.generation, actor: 'offline-operator')
  puts JSON.pretty_generate(row.audit.last)
when '--probe'
  snapshots = 100.times.map { |n| registry.pin(conversation_id: "ticket-#{n}") }
  abort 'candidate exposure remains' if registry.pointer.candidate || snapshots.any? { |p| p['bundle_id'] != row.stable }
  pinned = snapshots.first
  code = "import json,sys; from anchor_ai.serving_artifacts import pin_release_bundle; p=pin_release_bundle(json.loads(sys.argv[1]),tenant_id=1,required_contract='support-tools:v1',offline=True); print(json.dumps(dict(reply=p.complete(p.ticket),prompt=p.prompt,model=p.model,index=p.index,bundle_id=p.bundle_id,cache_namespace=p.cache_namespace)))"
  output, status = Open3.capture2('python3', '-c', code, JSON.generate(pinned), chdir: ReleaseLab::ROOT)
  abort 'snapshot consumer failed' unless status.success?
  result = JSON.parse(output)
  abort 'golden answer not restored' unless result.fetch('reply') == bundles.fetch('good').fetch('reply')
  puts output
  puts 'same ticket: prior known-good answer restored; all fixture cohorts restored; no app deploy or restart performed'
end
puts 'OFFLINE ONLY: live judge, resident model/index readiness, authenticated HTTP route and fleet propagation NOT verified; production deployment NOT approved.'
