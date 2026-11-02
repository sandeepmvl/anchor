require_relative '../../../labs/ai-product-ux/test_helper'

class AnswerFeedbackControllerTest < ActionDispatch::IntegrationTest
  include UXFixtures

  test 'verified ownership and server metadata' do
    message = stored
    first = feedback(message, tenant_id: '999', retrieved_chunk_ids: [999], prompt_version: 'evil')
    assert_equal 200, first.status
    first_ids = JSON.parse(first.body)
    second = feedback(message, comment: 'Changed customer@example.test')
    assert_equal first_ids, JSON.parse(second.body)
    assert_equal 1, GoldenCandidate.where(answer_message_id: message.id).count
    assert_equal 1, GoldenCandidate::FeedbackEvent.where(answer_message_id: message.id).count
    candidate = GoldenCandidate.find_by!(answer_message_id: message.id)
    assert_equal '1', candidate.tenant_id
    assert_equal [31], candidate.retrieved_chunk_ids
    assert_equal 'refund-ux:v1', candidate.prompt_version
    assert_equal 'conversation-a', candidate.conversation_id
    assert_equal 'queued', candidate.status
    assert_includes candidate.payload['comment'], '[EMAIL]'
    refute_includes candidate.payload.to_json, 'customer@example.test'
    refute_includes GoldenCandidate::FeedbackEvent.find_by!(answer_message_id: message.id).comment,
      'customer@example.test'
  end

  test 'wrong tenant customer conversation and signature are rejected' do
    message = stored
    [token(tenant: '2'), token(subject: 'customer-b'), token(conversation: 'conversation-b')].each do |auth|
      assert_equal 404, feedback(message, auth: auth).status
    end
    assert_equal 401, feedback(message, auth: token + 'forged').status
    assert_equal 0, GoldenCandidate.where(answer_message_id: message.id).count
    assert_equal 0, GoldenCandidate::FeedbackEvent.where(answer_message_id: message.id).count
  end

  test 'candidate failure rolls back feedback' do
    message = stored
    reject = ->(row) { row.errors.add(:base, 'forced test rejection') }
    GoldenCandidate.validate(reject)
    assert_raises(ActiveRecord::RecordInvalid) do
      GoldenCandidate.feedback!(claims: GoldenCandidate.claims!(token, purpose: 'answer_customer'),
        message_id: message.id, comment: 'bad')
    end
    assert_equal 0, GoldenCandidate::FeedbackEvent.where(answer_message_id: message.id).count
  ensure
    GoldenCandidate.skip_callback(:validate, :before, reject) if reject
  end

  test 'production storage fails without full redaction policy' do
    previous_env = Rails.env
    previous_policy = Rails.configuration.x.answer_redactor
    message = stored
    Rails.env = 'production'
    Rails.configuration.x.answer_redactor = nil
    assert_equal 422, feedback(message).status
    assert_equal 0, GoldenCandidate.where(answer_message_id: message.id).count
  ensure
    Rails.env = previous_env
    Rails.configuration.x.answer_redactor = previous_policy
  end

  test 'unset custom options preserve locally redacted values' do
    previous_policy = Rails.configuration.x.answer_redactor
    Rails.configuration.x.answer_redactor = ActiveSupport::OrderedOptions.new
    assert_equal 'Contact [EMAIL]', GoldenCandidate.safe!('Contact customer@example.test')
    assert_equal({'text' => '[EMAIL]'}, GoldenCandidate.safe!({'text' => 'customer@example.test'}))
  ensure
    Rails.configuration.x.answer_redactor = previous_policy
  end

  test 'configured redactor runs and invalid output rejects storage' do
    previous_policy = Rails.configuration.x.answer_redactor
    seen = []
    Rails.configuration.x.answer_redactor = ->(value) { seen << value; value }
    assert_equal '[EMAIL]', GoldenCandidate.safe!('customer@example.test')
    assert_equal ['[EMAIL]'], seen
    Rails.configuration.x.answer_redactor = ->(_value) { nil }
    message = stored
    assert_equal 422, feedback(message).status
    assert_equal 0, GoldenCandidate::FeedbackEvent.where(answer_message_id: message.id).count
    assert_equal 0, GoldenCandidate.where(answer_message_id: message.id).count
  ensure
    Rails.configuration.x.answer_redactor = previous_policy
  end
end
