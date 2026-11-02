require 'active_record'
require 'sqlite3'
require 'yaml'
require 'time'
require 'tmpdir'

module ReleaseLab
  ROOT = File.expand_path('../..', __dir__)
  def self.connect(path)
    ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: path, timeout: 2000)
    unless Object.const_defined?(:ApplicationRecord)
      Object.const_set(:ApplicationRecord, Class.new(ActiveRecord::Base) { self.abstract_class = true })
    end
    require File.join(ROOT, 'app/models/ai_release_pointer')
    require File.join(ROOT, 'app/services/ai_release_registry')
    require File.join(ROOT, 'db/migrate/20260101001600_create_ai_release_pointers')
    ActiveRecord::Migration.verbose = false
    CreateAiReleasePointers.new.migrate(:up) unless ActiveRecord::Base.connection.table_exists?(:ai_release_pointers)
  end

  def self.bundles
    YAML.safe_load(File.read(File.join(ROOT, 'config/model_releases.yml'))).fetch('release_bundles')
  end

  def self.registry(tenant = 1)
    AiReleaseRegistry.new(tenant_id: tenant, offline: true)
  end

  def self.health(row)
    { 'generation' => row.generation, 'candidate' => row.candidate,
      'control_count' => 16, 'candidate_count' => 4, 'control_quality' => 0.95,
      'candidate_quality' => 0.95, 'error_rate' => 0.0, 'p95_ms' => 100, 'window_seconds' => 600 }
  end
end
