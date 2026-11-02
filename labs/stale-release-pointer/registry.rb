require 'active_record'
require 'sqlite3'
require 'json'
require 'fileutils'
require 'time'

ROOT = File.expand_path('../..', __dir__)
WORK = File.join(ROOT, 'tmp/stale-release')
FileUtils.mkdir_p(WORK)
ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: File.join(WORK, 'registry.sqlite3'))
ActiveRecord::Schema.verbose = false
unless ActiveRecord::Base.connection.table_exists?(:ai_release_pointers)
  ActiveRecord::Schema.define do
    create_table :ai_release_pointers do |t|
      t.integer :tenant_id
      t.string :channel, default: 'reply'
      t.string :stable
      t.string :previous
      t.string :candidate
      t.integer :canary_percent, default: 0
      t.integer :generation, default: 0
      t.string :required_contract
      t.text :bundles
      t.text :audit
    end
    create_table :prompt_versions do |t|
      t.string :version
      t.text :template
      t.string :model_id
      t.float :temperature
      t.integer :max_tokens
    end
  end
end

# These fixture models exist only in this standalone process.
class AiReleasePointer < ActiveRecord::Base
  serialize :bundles, coder: JSON
  serialize :audit, coder: JSON
  after_initialize { self.audit ||= [] }
end
class PromptVersion < ActiveRecord::Base; end
require File.join(ROOT, 'app/services/prompt_registry')
require File.join(ROOT, 'app/services/ai_release_registry')

GOOD = { 'contract' => 'reply:v1', 'index' => 'kb-30d',
  'model' => 'offline-pinned', 'offline_only' => true, 'prompt' => 'chat:v1',
  'reply' => 'Refunds are allowed within 30 days.', 'ticket' => 'refund-20d' }.freeze
BAD = GOOD.merge('prompt' => 'chat:v2', 'reply' => 'Refunds are allowed only within seven days.').freeze
registry = AiReleaseRegistry.new(tenant_id: 1, offline: true)
command = ARGV.fetch(0)
case command
when 'inject'
  AiReleasePointer.delete_all
  row = AiReleaseRegistry.create!(tenant_id: 1, bundle: GOOD, contract: 'reply:v1')
  bad_id = AiReleaseRegistry.identity(BAD)
  row.update!(bundles: row.bundles.merge(bad_id => BAD), audit: [
    { 'action' => 'fixture_incomplete_rollback', 'app_version' => 'app-good',
      'registry_bundle' => bad_id, 'expected_bundle' => row.stable,
      'transaction_fields' => ['app_version'] }])
  puts JSON.generate(row.audit.last)
when 'pin'
  registry.with_request(conversation_id: 'refund-probe') do |pinned|
    puts JSON.generate(pinned.merge('provenance' => PromptRegistry.release_provenance))
  end
when 'inspect'
  row = registry.pointer
  puts JSON.generate(row.attributes.merge('liveness' => ActiveRecord::Base.connection.select_value('SELECT 1') == 1))
when 'resync'
  row = registry.pointer
  puts JSON.generate(registry.resync(expected_generation: row.generation, actor: 'drill-operator'))
when 'production'
  begin
    AiReleaseRegistry.new(tenant_id: 1).version_consistency!
    abort 'production unexpectedly authorized'
  rescue ArgumentError => error
    raise unless error.message == 'production_readiness_evidence_required'
    puts JSON.generate('production_blocked' => error.message)
  end
when 'readiness'
  begin
    puts JSON.generate(registry.version_consistency!)
  rescue ArgumentError => error
    puts JSON.generate('consistent' => false, 'error' => error.message)
  end
else
  abort 'unknown fixture command'
end
