if defined?(Rails::Application) && Rails.application
  require "rbconfig"

  class AiTaskRetryIsolationTest < ActiveSupport::TestCase
    test "durable task recovery exercise passes in an isolated process" do
      assert system(RbConfig.ruby, File.expand_path(__FILE__)),
        "Standalone task recovery tests failed"
    end
  end
else
  require_relative "../../labs/durable-background-jobs/test_support"

  class AiTaskRetryTest < Minitest::Test
    def setup
      DurableTaskLab.reset!
    end

    def test_worker_restart_resumes_without_duplicate_chunks
      task = DurableTaskLab.submit
      DurableTaskLab.worker(task.id, crash: true)
      assert_equal "running", task.reload.state
      assert_equal 9, task.progress
      assert_equal 10, LabEffect.count
      task.update!(lease_until: Time.current - 1)
      DurableTaskLab.worker(task.id)
      assert_equal "complete", task.reload.state
      assert_equal 500, task.progress
      assert_equal 500, LabEffect.count
      assert_equal 500, LabEffect.distinct.count(:effect_key)
      assert_equal 2, task.attempts
      puts "same large upload: worker restarted; complete 500/500; no duplicate chunks"
    end

    def test_checkpointed_agent_does_not_repeat_committed_tool_action
      task = AiTask.submit!(organization_id: 1, submission_key: "confirmed-refund",
        operation: "agent", payload: { "request" => { "approval_reference" => 7 },
          "execution_context" => "offline-signed-context", "request_id" => "request-7",
          "traceparent" => nil })
      DurableTaskLab.worker(task.id, crash: true)
      assert_equal 1, LabEffect.count
      assert_equal 0, task.reload.progress
      task.update!(lease_until: Time.current - 1)
      DurableTaskLab.worker(task.id)
      assert_equal "complete", task.reload.state
      assert_equal 1, LabEffect.where(kind: "confirmed_tool_action").count
      puts "checkpointed agent retry: committed tool action executed once"
    end

    def test_backoff_and_sanitized_terminal_failure
      task = DurableTaskLab.submit("failure", 1)
      failure = ->(*) { raise "secret credential and private document" }
      task.run!(ingestion: failure, agent: failure)
      assert_equal "retrying", task.reload.state
      assert_operator task.available_at, :>, Time.current
      task.run!(ingestion: failure, agent: failure)
      assert_equal 1, task.reload.attempts
      2.times do
        task.update!(available_at: Time.current - 1)
        task.run!(ingestion: failure, agent: failure)
      end
      assert_equal "dead_lettered", task.reload.state
      assert_equal 3, task.attempts
      assert_equal "task_failed", task.public_status.fetch(:error)
      refute_includes JSON.generate(task.public_status), "credential"
    end

    def test_missing_real_service_evidence_blocks_production
      error = assert_raises(AiTaskDispatcher::ReleaseBlocked) do
        AiTaskDispatcher.release_check!(production: true, evidence: {})
      end
      assert_equal "downstream_idempotency_evidence_missing", error.message
    end

    def test_finished_task_cannot_be_claimed_again
      task = DurableTaskLab.submit("once", 2)
      task.run!(ingestion: DurableTaskLab.ingestion, agent: DurableTaskLab.agent)
      task.run!(ingestion: ->(*) { flunk "completed task reran" }, agent: DurableTaskLab.agent)
      assert_equal 1, task.reload.attempts
      assert_equal 2, LabEffect.count
    end
  end
end
