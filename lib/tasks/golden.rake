namespace :golden do
  task :build, [:input, :tenant] => :environment do |_task, args|
    abort 'input and tenant required' unless args[:input] && args[:tenant]
    count = GoldenDatasets::Builder.new(tenant: args[:tenant]).call(args[:input])
    puts "Staged #{count} pending examples"
  end

  task :review, [:tenant, :reviewer, :ids] => :environment do |_task, args|
    abort 'tenant, reviewer and pipe-separated IDs required' unless args.values_at(:tenant, :reviewer, :ids).all?
    GoldenDatasets::Builder.new(tenant: args[:tenant]).approve!(
      keys: args[:ids].split('|'), reviewer: args[:reviewer])
    puts "Approved #{args[:ids]} by #{args[:reviewer]}"
  end

  task :version, [:tenant, :name, :against, :evidence, :tag] => :environment do |_task, args|
    abort 'Replay publication forbidden in production; use authenticated current evidence' if Rails.env.production?
    abort 'tenant, name, evidence and tag required' unless args.values_at(:tenant, :name, :evidence, :tag).all?
    docs = GoldenDatasets::Versioner.replay_evidence(args[:evidence], tenant: args[:tenant])
    snapshot = GoldenDatasets::Versioner.new(tenant: args[:tenant]).publish!(name: args[:name],
      against: args[:against].presence, docs: docs, built_against_tag: args[:tag])
    puts JSON.pretty_generate(snapshot.payload.slice('sha256', 'against', 'diff'))
    puts 'Replay maintenance verified; production authorization, release identity and model quality NOT verified.'
  end

  task :coverage, [:tenant] => :environment do |_task, args|
    puts JSON.pretty_generate(GoldenDatasets::Versioner.new(tenant: args[:tenant]).coverage)
  end

  task :revalidate, [:tenant, :evidence] => :environment do |_task, args|
    abort 'Replay evidence forbidden in production' if Rails.env.production?
    docs = GoldenDatasets::Versioner.replay_evidence(args[:evidence], tenant: args[:tenant])
    puts JSON.generate(needs_review: GoldenDatasets::Versioner.new(tenant: args[:tenant]).revalidate!(docs: docs))
  end

  task :inspect_replay, [:input] => :environment do |_task, args|
    File.foreach(args.fetch(:input)) do |line|
      row = JSON.parse(line)
      puts JSON.generate(row.slice('tenant_id', 'source_run_id', 'provenance', 'query'))
    end
  end

  task :promote_candidate, [:id, :correction] => :environment do |_task, args|
    abort 'candidate id and corrected example JSON required' unless args[:id] && args[:correction]
    token = ENV.fetch('ANCHOR_GOLDEN_REVIEW_CONTEXT')
    draft = GoldenDatasets::CandidateReviewer.new(token: token).promote!(
      candidate_id: args[:id], corrected: JSON.parse(File.read(args[:correction])))
    puts "Staged #{draft.key} as a pending draft; published snapshots unchanged"
  end
end
