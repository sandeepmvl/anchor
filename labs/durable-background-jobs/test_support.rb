require "active_record"
require "active_job"
require "action_controller/railtie"
require "action_controller/test_case"
require "active_support/testing/autorun"
require "tmpdir"
require "logger"
require "pathname"
require "json"

module DurableTaskLab
  ROOT = File.expand_path("../..", __dir__)
  WORKSPACE = Dir.mktmpdir("anchor-ai-tasks-")
  DB = File.join(WORKSPACE, "tasks.sqlite3")
  ActiveRecord::Base.establish_connection(adapter: "sqlite3", database: DB, timeout: 2000)
  ActiveRecord::Migration.verbose = false
end

# This isolated test process does not boot or change Anchor's application.
module Rails
  def self.root = Pathname.new(DurableTaskLab::ROOT)
  def self.env = ActiveSupport::EnvironmentInquirer.new("test")
  def self.logger = (@lab_logger ||= Logger.new(File::NULL))
end

class ApplicationRecord < ActiveRecord::Base
  self.abstract_class = true
end
class ApplicationJob < ActiveJob::Base; end
class ApplicationController < ActionController::Base; end
class Organization < ApplicationRecord; end
class Membership < ApplicationRecord; end
class LabEffect < ApplicationRecord; end
class DocumentIngestionClient
  class Error < StandardError; end
end

require File.join(DurableTaskLab::ROOT, "db/migrate/20260101000800_create_ai_tasks")
CreateAiTasks.new.migrate(:up)
ActiveRecord::Schema.define do
  create_table(:organizations) { |t| t.string :name }
  create_table(:memberships) { |t| t.integer :user_id; t.integer :organization_id }
  create_table(:lab_effects) { |t| t.string :effect_key; t.string :kind }
  add_index :lab_effects, :effect_key, unique: true
end
require File.join(DurableTaskLab::ROOT, "db/migrate/20260101001500_create_ai_operating_modes")
CreateAiOperatingModes.new.migrate(:up)
require File.join(DurableTaskLab::ROOT, "app/models/ai_operating_mode")
require File.join(DurableTaskLab::ROOT, "app/services/ai_incident_control")
require File.join(DurableTaskLab::ROOT, "app/models/ai_task")
require File.join(DurableTaskLab::ROOT, "app/services/ai_task_dispatcher")
require File.join(DurableTaskLab::ROOT, "app/jobs/ingest_document_job")
require File.join(DurableTaskLab::ROOT, "app/jobs/orchestrate_ticket_job")
require File.join(DurableTaskLab::ROOT, "app/controllers/api/ai_tasks_controller")
ActiveJob::Base.queue_adapter = :test

module DurableTaskLab
  def self.effect(key, kind)
    LabEffect.create_or_find_by!(effect_key: key) { |row| row.kind = kind }
  end

  def self.ingestion
    ->(_tenant, _reference, key) { effect(key, "chunk") }
  end

  def self.agent
    ->(_payload, key) { effect(key, "confirmed_tool_action") }
  end

  def self.submit(key = "upload-500", count = 500)
    AiTask.submit!(organization_id: 1, submission_key: key, operation: "ingestion",
      payload: { "references" => Array.new(count) { |i| "%032x" % i } })
  end

  def self.reset!
    AiTask.delete_all
    LabEffect.delete_all
    Membership.delete_all
    AiOperatingMode.delete_all
    Organization.delete_all
    Organization.create!(id: 1, name: "Durable task tenant")
    ActiveJob::Base.queue_adapter.enqueued_jobs.clear
  end

  def self.worker(task_id, crash: false)
    require "rbconfig"
    require "timeout"
    script = File.join(ROOT, "labs/durable-background-jobs/worker.rb")
    pid = Process.spawn(RbConfig.ruby, script, DB, task_id.to_s, crash ? "crash" : "finish",
      out: File::NULL, err: File::NULL)
    status = nil
    begin
      Timeout.timeout(20) { _, status = Process.wait2(pid) }
    rescue Timeout::Error
      Process.kill("KILL", pid)
      Process.wait(pid)
      raise "offline worker exceeded twenty seconds"
    end
    expected = crash ? 73 : 0
    raise "worker failed: #{status.inspect}" unless status.exitstatus == expected
  end
end

Minitest.after_run do
  ActiveRecord::Base.connection_pool.disconnect!
  FileUtils.remove_entry(DurableTaskLab::WORKSPACE)
end
