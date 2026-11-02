# This standalone harness owns an in-memory database and test doubles.
# Rails discovery must not install those doubles into the application process.
if $PROGRAM_NAME == __FILE__
require 'minitest/autorun'
require 'active_record'
require 'active_job'
require 'date'

ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: ':memory:')
ActiveRecord::Schema.define do
  create_table(:organizations) { |t| t.string :name }
end
class ApplicationRecord < ActiveRecord::Base
  self.abstract_class = true
end
class Organization < ApplicationRecord; end
class ApplicationJob < ActiveJob::Base; end
class PiiRedactionClient
  def self.value(payload)
    JSON.parse(JSON.generate(payload).gsub('customer@example.com', '[EMAIL]'))
  end
end
require_relative '../../db/migrate/20260101001400_create_answer_quality_samples'
CreateAnswerQualitySamples.new.migrate(:up)
require_relative '../../app/models/answer_quality_sample'
require_relative '../../app/jobs/score_answer_quality_job'
require_relative '../../app/jobs/orchestrate_ticket_job'
ActiveJob::Base.queue_adapter = :test

class OfflineQualityJob < ScoreAnswerQualityJob
  def score_sample(sample)
    raise 'raw evidence persisted' if sample.redacted_payload.to_json.include?('customer@example.com')
    {'status' => 'scored', 'score' => 0.0}
  end
end

class OnlineQualityMonitoringTest < Minitest::Test
  def setup
    AnswerQualitySample.delete_all
    Organization.delete_all
    ActiveJob::Base.queue_adapter.enqueued_jobs.clear
    @tenant = Organization.create!(name: 'Cedar')
    @other = Organization.create!(name: 'Birch')
  end

  def attributes(message = 'answer-1', organization = @tenant)
    {organization: organization, message_id: message, answer_version: 'v1',
     intent: 'refund', model_version: 'provider-v2', trace_id: 'trace-1',
     event_at: Time.now.utc,
     payload: {'answer' => 'customer@example.com is guaranteed a refund',
               'evidence' => [], 'citations' => []}}
  end

  def test_completion_is_async_idempotent_and_redacted
    sample = OrchestrateTicketJob.completed_answer(**attributes)
    assert_equal 'pending', sample.status
    assert_nil sample.score
    assert_equal 1, ActiveJob::Base.queue_adapter.enqueued_jobs.size
    refute_includes sample.redacted_payload.to_json, 'customer@example.com'
    assert_includes sample.redacted_payload.to_json, '[EMAIL]'
    assert_equal sample.id, OrchestrateTicketJob.completed_answer(**attributes).id
    assert_equal 1, AnswerQualitySample.count
    OfflineQualityJob.perform_now(@tenant.id, sample.id)
    OfflineQualityJob.perform_now(@tenant.id, sample.id)
    assert_equal 0.0, sample.reload.score
    assert_raises(ActiveRecord::RecordNotFound) do
      OfflineQualityJob.perform_now(@other.id, sample.id)
    end
  end

  def test_budget_reservations_and_authorized_feedback_association
    singleton = AnswerQualitySample.singleton_class
    original = AnswerQualitySample.method(:policy)
    singleton.define_method(:policy) { original.call.merge('daily_tenant_limit' => 1) }
    sample = AnswerQualitySample.capture!(**attributes)
    assert_nil AnswerQualitySample.capture!(**attributes('answer-2'))
    assert AnswerQualitySample.capture!(**attributes('answer-3', @other))
    # Match GoldenCandidate::FeedbackEvent's persisted tenant field.
    feedback = Struct.new(:id, :attributes).new(42, {'tenant_id' => @other.id.to_s})
    AnswerQualitySample.attach_feedback!(feedback: feedback, message_id: sample.message_id)
    assert_nil sample.reload.feedback_event_id
    feedback.attributes['tenant_id'] = @tenant.id.to_s
    AnswerQualitySample.attach_feedback!(feedback: feedback, message_id: sample.message_id)
    assert_equal 42, sample.reload.feedback_event_id
    assert_equal 'v1', sample.answer_version
    assert_nil sample.score
    feedback.attributes['organization_id'] = @other.id
    assert_raises(ActiveRecord::RecordNotFound) do
      AnswerQualitySample.attach_feedback!(feedback: feedback, message_id: sample.message_id)
    end
    assert_equal 42, sample.reload.feedback_event_id
  ensure
    singleton.define_method(:policy, original) if original
  end

  def test_missing_real_transport_configuration_refuses_scoring
    sample = AnswerQualitySample.capture!(**attributes)
    secret = ENV.delete('ANCHOR_AI_SECRET')
    assert_raises(KeyError) { ScoreAnswerQualityJob.new.perform(@tenant.id, sample.id) }
    assert_equal 'pending', sample.reload.status
  ensure
    ENV['ANCHOR_AI_SECRET'] = secret if secret
  end
end
end # standalone harness
