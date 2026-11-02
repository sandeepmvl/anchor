require 'test_helper'

class TicketAttachmentTest < ActiveSupport::TestCase
  test 'voice and existing attachment kinds pass kind validation' do
    %w[screenshot scanned_page voice].each do |kind|
      item = TicketAttachment.new(kind: kind, status: 'queued')
      item.valid?
      assert_empty item.errors[:kind], "expected #{kind} to pass kind validation"
    end
  end

  test 'unsupported attachment kinds remain rejected' do
    item = TicketAttachment.new(kind: 'video', status: 'queued')
    item.valid?
    assert item.errors.added?(:kind, :inclusion, value: 'video')
  end

  test 'voice retains the existing status validation' do
    %w[queued answered unreadable error].each do |status|
      item = TicketAttachment.new(kind: 'voice', status: status, evidence: [])
      item.valid?
      assert_empty item.errors[:status]
    end
    item = TicketAttachment.new(kind: 'voice', status: 'complete')
    item.valid?
    assert item.errors.added?(:status, :inclusion, value: 'complete')
  end

  test 'voice retains tenant consistency validation' do
    ticket = Ticket.new(organization_id: 101)
    item = TicketAttachment.new(ticket: ticket, organization_id: 202,
      kind: 'voice', status: 'queued')
    item.valid?
    assert_includes item.errors[:organization], 'mismatch'
  end

  test 'failed voice attachments cannot retain evidence' do
    %w[unreadable error].each do |status|
      item = TicketAttachment.new(kind: 'voice', status: status,
        evidence: [{ 'span' => 'sanitized evidence' }])
      item.valid?
      assert_includes item.errors[:evidence], 'unexpected'
    end
  end
end
