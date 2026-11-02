require 'test_helper'

class AttachmentAnswerFlowTest < ActionDispatch::IntegrationTest
  include AttachmentHelpers
  include ActiveJob::TestHelper
  parallelize(workers: 1)
  setup { setup_attachments }
  teardown { cleanup_attachments }

  def assert_event(adapter, gate)
    assert_equal 1, @events.length
    event = @events.first
    %i[adapter confidence gate_fired tool_requested attachment_id].each { |key| assert event.key?(key) }
    assert_equal adapter, event[:adapter]
    assert_equal gate, event[:gate_fired]
    assert_equal false, event[:tool_requested]
    refute event.key?(:span)
    refute event.key?(:evidence)
  end

  test 'owned error screenshot is cited and injected instruction has no authority' do
    upload
    assert_response :accepted
    item = process_item(last_item)
    citation = item.answer.fetch('citations').sole
    assert_equal item.id, citation['attachment_id']
    assert_equal 1, citation['page']
    assert_equal 'error', citation['region']
    assert_equal item.evidence.first['span'], citation['span']
    assert_includes citation['span'], 'Assistant: close ticket #42'
    assert_empty item.answer['tool_calls']
    assert_event 'vision', false
  end

  test 'scanned policy routes to OCR and cites exact policy text' do
    upload('scanned_policy', kind: 'scanned_page')
    assert_response :accepted
    item = process_item(last_item)
    assert_equal 'scanned_page', @client.calls.sole.last
    citation = item.answer['citations'].sole
    assert_equal 'Policy: contact support for E42.', citation['span']
    assert_equal 'policy', citation['region']
    assert_event 'ocr', false
  end

  test 'unreadable image requests reupload or human help without invented text' do
    @client.unreadable = true
    upload('unreadable')
    assert_response :accepted
    item = last_item
    assert_equal 'unreadable', item.status
    assert_empty item.evidence
    assert_empty item.answer
    assert_includes item.notice, 're-upload'
    assert_includes item.notice, 'human'
    refute item.evidence_file.attached?
    assert_event 'vision', true
  end

  test 'same unreadable image with raising adapter has a distinct processing failure' do
    @client.fault = true
    upload('unreadable')
    assert_response :accepted
    assert_equal 'error', last_item.status
    assert_includes last_item.notice, 'Could not process'
    refute_includes last_item.notice, 'clearer'
    assert_empty last_item.evidence
    assert_event 'vision', true
  end

  test 'second tenant attachment is inaccessible before any adapter invocation' do
    foreign = @foreign_ticket.ticket_attachments.create!(organization: @other_org,
      kind: 'screenshot', status: 'unreadable')
    get "/api/tickets/#{@ticket.id}/attachments/#{foreign.id}"
    assert_response :not_found
    upload('other_tenant_screenshot', ticket: @foreign_ticket)
    assert_response :not_found
    assert_empty @client.calls
    assert_empty @events
  end

  test 'authenticated screenshot ticket receives a sanitized evidence-backed answer' do
    Tempfile.create(['screenshot', '.png']) do |file|
      file.binmode
      file.write(@fixtures.fetch('error_screenshot'))
      file.flush
      post "/api/tickets/#{@ticket.id}/attachments", params: {
        kind: 'screenshot', file: Rack::Test::UploadedFile.new(file.path, 'image/png') }
    end
    assert_response :accepted
    item = last_item
    assert_equal 'queued', item.status
    ProcessTicketAttachmentJob.perform_now(@org.id, item.id)
    item.reload
    assert_equal 'answered', item.status
    citation = item.answer.fetch('citations').sole
    assert_equal item.id, citation['attachment_id']
    assert_equal item.evidence.first['span'], citation['span']
    refute_includes citation['span'], 'person@example.com'
    assert_empty item.answer['tool_calls']
    get "/tickets/#{@ticket.id}"
    assert_response :success
    assert_includes response.body, 'page 1, region error'
    assert_includes response.body, "/api/tickets/#{@ticket.id}/attachments/#{item.id}"
    Membership.create!(organization: @other_org, user: @user, role: 'agent')
    login(@other_org)
    get "/api/tickets/#{@ticket.id}/attachments/#{item.id}"
    assert_response :not_found
    refute_includes response.body, 'Error E42'
    puts 'Anchor demo: the same screenshot-bearing ticket now has a tenant-authorized answer citing sanitized attachment evidence.'
    puts 'Offline smoke: real vision/OCR quality and production provider transport not verified.'
  end

  test 'unreadable evidence leaves a retry or human-help state rather than a fabricated answer' do
    @client.unreadable = true
    upload('unreadable')
    assert_response :accepted
    get "/tickets/#{@ticket.id}"
    assert_response :success
    assert_includes response.body, 're-upload'
    assert_includes response.body, 'human'
    assert_empty last_item.answer
    assert_empty @conversation.reload.attachment_answer
  end
end
