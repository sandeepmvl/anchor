require 'active_record'
require 'minitest/autorun'
require 'tempfile'
require 'json'
require_relative '../../app/models/golden_dataset'
require_relative '../../app/services/ai_telemetry'
require_relative '../../app/services/golden_datasets/builder'
require_relative '../../app/services/golden_datasets/versioner'
require_relative '../../db/migrate/20260101000500_create_golden_datasets'

unless defined?(Rails) && Rails.respond_to?(:application) && Rails.application
  ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: ':memory:')
  ActiveRecord::Migration.verbose = false
  CreateGoldenDatasets.new.migrate(:up)
end

class GoldenTestCase < Minitest::Test
  ROOT = File.expand_path('../..', __dir__)
  TENANT = 'tenant-demo'

  def setup
    # Rails may prepare or replace the test schema after loading this file.
    # Install guards on the actual database used by each test, before its transaction.
    CreateGoldenDatasets.new.install_guards!(GoldenDataset.connection)
    guards = GoldenDataset.connection.select_values(
      "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'golden_datasets'")
    %w[golden_version_no_update golden_version_no_delete golden_kind_no_change].each do |name|
      raise "missing golden database guard: #{name}" unless guards.include?(name)
    end
    GoldenDataset.connection.begin_transaction(joinable: false)
    @events = []
    @previous_sink = AiTelemetry.sink
    AiTelemetry.sink = ->(event) { @events << event }
    @builder = GoldenDatasets::Builder.new(tenant: TENANT)
    @versioner = GoldenDatasets::Versioner.new(tenant: TENANT)
  end

  def teardown
    GoldenDataset.connection.rollback_transaction
    AiTelemetry.sink = @previous_sink
  end

  def fixture(name)
    File.join(ROOT, 'eval/datasets', name)
  end

  def replay
    fixture('release_query_replay.jsonl')
  end

  def docs
    GoldenDatasets::Versioner.replay_evidence(replay, tenant: TENANT)
  end

  def baseline
    row = JSON.parse(File.foreach(fixture('golden_v1.jsonl')).first)
    GoldenDataset.create!(tenant_id: TENANT, kind: 'example', key: row['source_run_id'], payload: row.except('tenant_id'))
    @versioner.publish!(name: 'v1', against: nil, docs: docs, built_against_tag: 'test-release')
  end

  def with_rows(rows)
    Tempfile.create(['golden', '.jsonl']) do |file|
      file.write(rows.map { |row| JSON.generate(row) + "\n" }.join)
      file.flush
      yield file.path
    end
  end
end
