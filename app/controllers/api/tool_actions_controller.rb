class Api::ToolActionsController < ApplicationController
  # Validate the original JSON envelope, without Rails-generated wrapper keys.
  wrap_parameters false

  def create
    membership = TicketAccessPolicy.new(session).authenticated_membership
    body = request.request_parameters
    allowed = %w[action arguments approval_receipt idempotency_key]
    raise ToolAuthorizationPolicy::Denied, 'invalid_arguments' unless (body.keys - allowed).empty?
    action = body['action']
    arguments = body['arguments']
    result = Membership.transaction do
      # Re-read current role and organization; serialize against membership changes.
      membership.lock!
      policy = ToolAuthorizationPolicy.new(membership)
      order = policy.authorize!(action, arguments)
      if action == 'issue_refund'
        policy.approval!(body['approval_receipt'], order,
                         arguments['amount_cents'], body['idempotency_key'])
      end
      execute!(action, order, arguments, body['idempotency_key'])
    end
    render json: { ok: true, data: result }
  rescue TicketAccessPolicy::Unauthenticated, ActiveRecord::RecordNotFound
    render json: { ok: false, error: 'unauthenticated' }, status: :unauthorized
  rescue ToolAuthorizationPolicy::Denied => error
    render json: { ok: false, error: error.message }, status: :forbidden
  end

  private

  def execute!(action, order, arguments, key)
    case action
    when 'lookup_order'
      { order_id: order.reference, payment_id: order.payment_id,
        status: order.status, customer_contact: '[REDACTED]' }
    when 'check_refund'
      { order_id: order.reference, eligible: order.refund_eligible,
        reason: order.refund_eligible ? 'within_window' : 'outside_window' }
    when 'issue_refund'
      amount = arguments.fetch('amount_cents')
      if order.refund_key
        unless order.refund_key == key && order.refunded_cents == amount
          raise ToolAuthorizationPolicy::Denied, 'idempotency_conflict'
        end
      else
        unless order.refund_eligible && amount <= order.refund_limit_cents
          raise ToolAuthorizationPolicy::Denied, 'refund_ineligible'
        end
        order.update!(refund_key: key, refunded_cents: amount, status: 'refunded')
      end
      { order_id: order.reference, amount_cents: amount, status: 'refunded' }
    else
      raise ToolAuthorizationPolicy::Denied, 'action_denied'
    end
  end
end
