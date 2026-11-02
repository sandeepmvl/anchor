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
        receipt = body['approval_receipt']
        executor = ToolActionExecutor.new
        if receipt.is_a?(String) && receipt.start_with?('safe:')
          executor.execute(membership, order, arguments, body['idempotency_key'], receipt)
        else
          policy.approval!(receipt, order, arguments['amount_cents'], body['idempotency_key'])
          executor.execute_approved(membership, order, arguments, body['idempotency_key'])
        end
      else
        execute!(action, order, arguments, body['idempotency_key'])
      end
    end
    render json: { ok: true, data: result }
  rescue TicketAccessPolicy::Unauthenticated, ActiveRecord::RecordNotFound
    render json: { ok: false, error: 'unauthenticated' }, status: :unauthorized
  rescue ToolAuthorizationPolicy::Denied, ToolActionExecutor::Denied => error
    render json: { ok: false, error: error.message }, status: :forbidden
  end

  def preview
    safe_request(:preview)
  end

  def confirm
    safe_request(:confirm)
  end

  def reverse
    safe_request(:reverse)
  end

  private

  def safe_request(mode)
    member = TicketAccessPolicy.new(session).authenticated_membership
    body = request.request_parameters
    result = Membership.transaction do
      member.lock!
      executor = ToolActionExecutor.new
      if mode == :reverse
        raise ToolActionExecutor::Denied, 'invalid_arguments' unless body.keys == ['tool_action_id']
        action = ToolAction.find_by!(id: body['tool_action_id'], organization_id: member.organization_id)
        ToolActionReverser.new.reverse(member, action)
      else
        allowed = mode == :preview ? %w[action arguments] : %w[action arguments proposal_receipt idempotency_key]
        unless (body.keys - allowed).empty? && body['action'] == 'issue_refund'
          raise ToolActionExecutor::Denied, 'invalid_arguments'
        end
        order = ToolAuthorizationPolicy.new(member).authorize!(body['action'], body['arguments'])
        if mode == :preview
          executor.preview(member, order, body['arguments'])
        else
          # The signed proposal, not submitted arguments, supplies confirmation state.
          claims = AgentApproval.safe_claims(body['proposal_receipt'])
          unless claims && claims['arguments'] == body['arguments']
            raise ToolActionExecutor::Denied, 'proposal_mismatch'
          end
          executor.confirm(member, order, body['proposal_receipt'], body['idempotency_key'])
        end
      end
    end
    render json: { ok: true, data: result }
  rescue TicketAccessPolicy::Unauthenticated, ActiveRecord::RecordNotFound
    render json: { ok: false, error: 'unauthenticated' }, status: :unauthorized
  rescue ToolAuthorizationPolicy::Denied, ToolActionExecutor::Denied => error
    render json: { ok: false, error: error.message }, status: :forbidden
  end

  def execute!(action, order, arguments, key)
    case action
    when 'lookup_order'
      { order_id: order.reference, payment_id: order.payment_id,
        status: order.status, customer_contact: '[REDACTED]' }
    when 'check_refund'
      { order_id: order.reference, eligible: order.refund_eligible,
        reason: order.refund_eligible ? 'within_window' : 'outside_window' }
    when 'issue_refund'
      # Refund writes must pass the public action's approval and audited executor.
      raise ToolActionExecutor::Denied, 'confirmation_required'
    else
      raise ToolAuthorizationPolicy::Denied, 'action_denied'
    end
  end
end
