if ENV['ANCHOR_AUTHZ_OFFLINE'] == '1'
  require_relative '../../labs/scoped-agent-authorization/test_support'
else
  require 'test_helper'
end

class AuthorizationProbeController < Api::ToolActionsController
  attr_reader :executor_calls
  private
  def execute!(*arguments)
    @executor_calls = (@executor_calls || 0) + 1
    super
  end
end

class ToolAuthorizationTest < ActionController::TestCase
  tests AuthorizationProbeController

  setup do
    @routes = ActionDispatch::Routing::RouteSet.new
    @routes.draw do
      post '/api/tool_actions', to: 'authorization_probe#create'
      post '/api/tool_actions/preview', to: 'authorization_probe#preview'
      post '/api/tool_actions/confirm', to: 'authorization_probe#confirm'
    end
    suffix = SecureRandom.hex(6)
    @local = Organization.create!(name: "Authz local #{suffix}")
    @foreign = Organization.create!(name: "Authz foreign #{suffix}")
    # Use the real password setter to satisfy both application and offline constraints.
    user = User.create!(email: "authz-#{suffix}@example.test", password: 'Authz-fixture-password-3740')
    @user_id = user.id
    @actor = Membership.create!(organization: @local, user_id: @user_id, role: 'agent')
    @order = Order.create!(organization: @local, reference: 'A104', payment_id: 'offline_local')
    @other = Order.create!(organization: @foreign, reference: 'B204', payment_id: 'offline_foreign')
    @request.session[:membership_id] = @actor.id
    @request.session[:user_id] = @user_id
  end

  def receipt(order = @order, key = 'refund_1')
    ToolAuthorizationPolicy.approval_receipt(membership: @actor, order: order,
      amount_cents: 100, idempotency_key: key)
  end

  def proposal(order = @order)
    { action: 'issue_refund', arguments: { order_id: order.reference, amount_cents: 100 },
      approval_receipt: receipt(order), idempotency_key: 'refund_1' }
  end

  def submit(body)
    post :create, body: body.to_json, as: :json
  end

  def assert_blocked(body, status = 403)
    submit(body)
    assert_response status
    assert_equal 0, @controller.executor_calls.to_i
    assert_equal 0, @order.reload.refunded_cents
    assert_equal 0, @other.reload.refunded_cents
  end

  test 'approved local refund reaches executor and retry does not double refund' do
    body = proposal.except(:approval_receipt)
    post :preview, body: body.slice(:action, :arguments).to_json, as: :json
    assert_response :success
    preview = response.parsed_body.fetch('data')
    assert_equal true, preview.fetch('dry_run')
    assert_equal 0, @order.reload.refunded_cents

    post :confirm, body: body.merge(
      proposal_receipt: preview.fetch('proposal_receipt')).to_json, as: :json
    assert_response :success
    confirmation = response.parsed_body.fetch('data')
    body[:approval_receipt] = confirmation.fetch('approval_receipt')
    action = ToolAction.find(confirmation.fetch('tool_action_id'))
    assert_equal 'confirmed', action.status
    assert_equal 0, @order.reload.refunded_cents

    submit(body)
    assert_response :success
    first_result = response.parsed_body.fetch('data')
    assert_equal 'executed', action.reload.status
    assert_equal 1, action.action_reversal_logs.where(event: 'execution').count
    assert_equal 100, @order.reload.refunded_cents
    submit(body)
    assert_response :success
    assert_equal first_result, response.parsed_body.fetch('data')
    assert_equal 100, @order.reload.refunded_cents
    assert_equal 1, action.action_reversal_logs.where(event: 'execution').count
    assert_equal 0, @other.reload.refunded_cents
    puts 'PASS local confirmed refund: audited executor reached once; repeat-safe ledger'
  end

  test 'foreign order with otherwise valid approval stops before executor' do
    assert_blocked(proposal(@other))
    assert_equal 'resource_denied', response.parsed_body['error']
    puts 'PASS foreign approved refund: executor calls = 0'
  end

  test 'forged tenant and actor fields cannot establish scope' do
    body = proposal(@other)
    body[:arguments][:tenant_id] = @foreign.id
    body[:arguments][:actor_id] = @actor.id
    assert_blocked(body)
    assert_equal 'invalid_arguments', response.parsed_body['error']
    puts 'PASS forged tenant and actor: executor calls = 0'
  end

  test 'end customer cannot acquire agent permission' do
    body = proposal
    @actor.update!(role: 'end_customer')
    assert_blocked(body)
    assert_equal 'role_denied', response.parsed_body['error']
    puts 'PASS current end-customer role: executor calls = 0'
  end

  test 'missing authenticated session fails closed despite claimed identity' do
    body = proposal.merge(tenant_id: @local.id, actor_id: @actor.id)
    @request.session.clear
    assert_blocked(body, 401)
    puts 'PASS unauthenticated internal request: executor calls = 0'
  end

  test 'missing approval refuses release to executor' do
    assert_blocked(proposal.except(:approval_receipt))
    assert_equal 'approval_required', response.parsed_body['error']
  end

  test 'expired approval refuses release to executor' do
    token = Rails.application.message_verifier('tool-approval').generate(
      {}, purpose: ToolAuthorizationPolicy.settings.fetch('approval_purpose'), expires_in: -1)
    assert_blocked(proposal.merge(approval_receipt: token))
    assert_equal 'approval_required', response.parsed_body['error']
  end

  test 'unknown action has no executor policy' do
    assert_blocked(proposal.merge(action: 'delete_order'))
    assert_equal 'action_denied', response.parsed_body['error']
  end
end
