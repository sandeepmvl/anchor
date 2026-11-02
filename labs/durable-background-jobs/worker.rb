require "active_record"
require "active_support/all"
require "logger"
module Rails
  def self.logger = (@logger ||= Logger.new(File::NULL))
end
class ApplicationRecord < ActiveRecord::Base
  self.abstract_class = true
end
class LabEffect < ApplicationRecord; end
ActiveRecord::Base.establish_connection(adapter: "sqlite3", database: ARGV.fetch(0), timeout: 2000)
require_relative "../../app/models/ai_task"
task = AiTask.find(ARGV.fetch(1))
adapter = lambda do |key, kind|
  LabEffect.create_or_find_by!(effect_key: key) { |row| row.kind = kind }
end
ingestion = ->(_tenant, _reference, key) { adapter.call(key, "chunk") }
agent = ->(_payload, key) { adapter.call(key, "confirmed_tool_action") }
hook = if ARGV.fetch(2) == "crash"
  ->(index) { exit! 73 if index == [9, task.total - 1].min }
end
task.run!(ingestion: ingestion, agent: agent, after_unit: hook)
raise "task did not complete" unless task.reload.state == "complete"
