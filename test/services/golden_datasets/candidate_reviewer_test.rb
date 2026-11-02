require_relative '../../../labs/ai-product-ux/test_helper'

class CandidateReviewerTest < ActiveSupport::TestCase
  include UXFixtures

  def candidate
    message = stored
    GoldenCandidate.feedback!(claims: GoldenCandidate.claims!(token, purpose: 'answer_customer'),
      message_id: message.id, comment: 'Please correct this.').last
  end

  def correction
    raw = JSON.parse(File.foreach(Rails.root.join('eval/datasets/golden_v1.jsonl')).first)
    raw['tenant_id'] = '1'
    raw.fetch('context').each { |doc| doc['tenant_id'] = '1' }
    raw
  end

  test 'only authorized tenant reviewer can promote' do
    row = candidate
    assert_raises(GoldenCandidate::AuthorizationError) do
      GoldenDatasets::CandidateReviewer.new(token: token)
    end
    assert_raises(GoldenCandidate::AuthorizationError) do
      GoldenDatasets::CandidateReviewer.new(token: token(reviewer: true, role: 'customer'))
    end
    assert_raises(ActiveRecord::RecordNotFound) do
      GoldenDatasets::CandidateReviewer.new(token: token(tenant: '2', reviewer: true)).promote!(
        candidate_id: row.id, corrected: {})
    end
    assert_equal 'queued', row.reload.status
  end

  test 'review creates one pending draft without changing published snapshots' do
    row = candidate
    snapshots = GoldenDataset.where(kind: 'snapshot').order(:id).map(&:attributes)
    auth = token(reviewer: true, subject: 'reviewer-a')
    draft = GoldenDatasets::Builder.new(tenant: '1').promote_candidate!(
      candidate_id: row.id, corrected: correction, token: auth)
    assert_equal 'pending', draft.payload['status']
    assert_equal 'pending', draft.payload['review_status']
    assert_equal 'reviewer-a', draft.payload['reviewed_by']
    assert_equal [31], draft.payload['retrieved_chunk_ids']
    assert_equal 'promoted', row.reload.status
    again = GoldenDatasets::CandidateReviewer.new(token: auth).promote!(
      candidate_id: row.id, corrected: correction)
    assert_equal draft.id, again.id
    assert_equal snapshots, GoldenDataset.where(kind: 'snapshot').order(:id).map(&:attributes)
    puts 'Feedback reviewed into a pending draft; published snapshots, including v2, unchanged.'
  end
end
