require 'test_helper'

class ProcessTicketAttachmentJobTest < ActionDispatch::IntegrationTest
  include AttachmentHelpers
  include ActiveJob::TestHelper
  parallelize(workers: 1)
  setup { setup_attachments }
  teardown { cleanup_attachments }

  test 'upload persists only sanitized evidence and queues only identifiers' do
    clear_enqueued_jobs
    upload
    assert_response :accepted
    item = last_item
    job = enqueued_jobs.find { |entry| entry[:job] == ProcessTicketAttachmentJob }
    assert_equal [@org.id, item.id], job[:args]
    assert item.evidence_file.attached?
    bytes = item.evidence_file.download
    assert_equal item.evidence, JSON.parse(bytes)
    refute_includes bytes, 'person@example.com'
    refute bytes.start_with?("\x89PNG".b)
    assert_includes bytes, '[REDACTED_EMAIL]'
    assert_equal 1, ActiveStorage::Attachment.where(record: item).count
    assert_equal 'application/json', item.evidence_file.blob.content_type
  end

  test 'duplicate deliveries are idempotent and wrong organization cannot process' do
    upload
    item = last_item
    process_item(item)
    assert_equal 'answered', item.status
    answer = item.answer.deep_dup
    process_item(item)
    assert_equal answer, item.answer
    assert_raises(ActiveRecord::RecordNotFound) do
      ProcessTicketAttachmentJob.perform_now(@other_org.id, item.id)
    end
    assert_equal answer, item.reload.answer
  end

  test 'failed extraction records a visible failure without raw persistence or queued answer' do
    clear_enqueued_jobs
    @client.fault = true
    upload('unreadable')
    assert_response :accepted
    assert_equal 'error', last_item.status
    assert_includes last_item.notice, 'Could not process'
    assert_empty last_item.evidence
    refute last_item.evidence_file.attached?
    refute enqueued_jobs.any? { |entry| entry[:job] == ProcessTicketAttachmentJob }
  end
end
