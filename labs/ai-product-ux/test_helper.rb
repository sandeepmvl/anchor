require_relative '../../test/test_helper'

module UXFixtures
  def token(tenant: '1', subject: 'customer-a', conversation: 'conversation-a', reviewer: false, role: nil)
    purpose = reviewer ? 'golden_reviewer' : 'answer_customer'
    GoldenCandidate.verifier.generate({'tenant_id' => tenant, 'subject' => subject,
      'conversation_id' => conversation, 'role' => role || (reviewer ? 'golden_reviewer' : 'customer'),
      'expires_at' => Time.now.to_i + 600}, purpose: purpose)
  end

  def stored(tenant: '1', subject: 'customer-a', conversation: 'conversation-a')
    GoldenCandidate::AnswerMessage.create!(tenant_id: tenant, customer_id: subject,
      conversation_id: conversation, prompt_version: 'refund-ux:v1', retrieved_chunk_ids: [31],
      payload: {'question' => 'When will my refund arrive?', 'cited_answer' => {
        'answer' => 'Refunds require review.', 'decision' => 'hedge', 'citations' => [
          {'chunk_id' => 31, 'quoted_span' => 'Refunds require review.'}]}})
  end

  def feedback(message, auth: token, **extra)
    post '/api/answer_feedback', params: {message_id: message.id, rating: 'down',
      comment: 'Email me at customer@example.test'}.merge(extra),
      headers: {'Authorization' => "Bearer #{auth}"}, as: :json
    response
  end
end
