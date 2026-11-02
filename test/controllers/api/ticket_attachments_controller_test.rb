require 'test_helper'

class Api::TicketAttachmentsControllerTest < ActionDispatch::IntegrationTest
  include AttachmentHelpers
  parallelize(workers: 1)
  setup { setup_attachments }
  teardown { cleanup_attachments }

  test 'session is required and password login checks membership' do
    delete '/session'
    upload
    assert_response :unauthorized
    assert_empty @client.calls
    post '/session', params: { email: @user.email, password: PASSWORD, organization_id: @other_org.id }
    assert_response :unauthorized
  end

  test 'admin and agent can upload only inside their organization' do
    %w[admin agent].each do |role|
      @membership.update!(role: role)
      upload
      assert_response :accepted
      before = @client.calls.length
      upload('other_tenant_screenshot', ticket: @foreign_ticket,
        extra: { organization_id: @org.id, tenant_id: @other_org.id })
      assert_response :not_found
      assert_equal before, @client.calls.length
    end
  end

  test 'end customers access only their own tickets' do
    @membership.update!(role: 'end_customer')
    upload
    assert_response :accepted
    another = User.create!(email: "customer-#{SecureRandom.hex(4)}@example.test", password: PASSWORD)
    ticket = Ticket.create!(organization: @org, user: another, subject: 'Not yours')
    upload(ticket: ticket)
    assert_response :not_found
    assert_equal 1, @client.calls.length
  end

  test 'foreign attachment identifiers and forged claims cannot disclose evidence' do
    foreign = @foreign_ticket.ticket_attachments.create!(organization: @other_org,
      kind: 'screenshot', status: 'unreadable', notice: 'Private notice')
    get "/api/tickets/#{@ticket.id}/attachments/#{foreign.id}", params: { tenant_id: @other_org.id }
    assert_response :not_found
    refute_includes response.body, 'Private notice'
    assert_empty @client.calls
  end

  test 'attachment operations retain CSRF protection' do
    old = ActionController::Base.allow_forgery_protection
    ActionController::Base.allow_forgery_protection = true
    upload
    assert_response :unprocessable_entity
    assert_empty @client.calls
  ensure
    ActionController::Base.allow_forgery_protection = old
  end
end
