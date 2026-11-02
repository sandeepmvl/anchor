class ToolActionExecutor
  class Denied < StandardError; end
  TTL = 300

  def initialize(verifier: Rails.application.message_verifier(AgentApproval::SAFE_ACTION_PURPOSE),
                 local: Rails.env.development? || Rails.env.test?, clock: -> { Time.current })
    @verifier, @local, @clock = verifier, local, clock
  end

  def authorize!(member, order, reverse: false)
    if !reverse && AiIncidentControl.human_only?(order.organization_id)
      raise Denied, 'human_only'
    end
    roles = reverse ? %w[admin] : %w[admin agent]
    unless member.organization_id == order.organization_id && roles.include?(member.role)
      raise Denied, 'action_denied'
    end
  end

  def preview(member, order, arguments)
    authorize!(member, order)
    amount = arguments['amount_cents']
    unless arguments.keys.sort == %w[amount_cents order_id] &&
           arguments['order_id'] == order.reference && amount.is_a?(Integer) &&
           amount.positive? && order.refund_eligible &&
           amount <= order.refundable_balance_cents && order.refund_key.nil?
      raise Denied, 'refund_ineligible'
    end
    claims = { 'kind' => 'proposal', 'organization_id' => order.organization_id,
               'actor_id' => member.user_id, 'order_id' => order.id,
               'arguments' => arguments, 'before' => order.safe_action_state,
               'exp' => @clock.call.to_i + TTL }
    { dry_run: true, tier: 'T3', backend: 'local_fake_payment',
      before_balance_cents: order.refundable_balance_cents,
      after_balance_cents: order.refundable_balance_cents - amount,
      proposal_receipt: sign(claims), expires_at: claims['exp'] }
  end

  def confirm(member, order, receipt, key)
    authorize!(member, order)
    claims = verify(receipt, 'proposal', member, order)
    raise Denied, 'idempotency_required' unless key.is_a?(String) && key.match?(/\A[A-Za-z0-9_-]{8,200}\z/)
    action = ToolAction.transaction do
      order.lock!
      previous = ToolAction.find_by(organization_id: order.organization_id, idempotency_key: key)
      if previous
        exact!(previous, member, order, claims['arguments'])
        previous
      else
        raise Denied, 'stale_proposal' unless order.safe_action_state == claims['before']
        ToolAction.create!(organization_id: order.organization_id, actor_id: member.user_id,
                           order: order, idempotency_key: key, arguments: claims['arguments'],
                           before_state: claims['before'], expires_at: Time.at(claims['exp']))
      end
    end
    { tool_action_id: action.id, approval_receipt: sign({
      'kind' => 'confirmation', 'organization_id' => action.organization_id,
      'actor_id' => action.actor_id, 'order_id' => action.order_id,
      'arguments' => action.arguments, 'action_id' => action.id,
      'exp' => action.expires_at.to_i }) }
  end

  def execute(member, order, arguments, key, receipt)
    authorize!(member, order)
    claims = verify(receipt, 'confirmation', member, order)
    raise Denied, 'proposal_mismatch' unless claims['arguments'] == arguments
    raise Denied, 'payment_adapter_required' unless @local
    AiIncidentControl.with_ai_permission!(order.organization_id) do
    ToolAction.transaction do
      order.lock!
      action = ToolAction.lock.find(claims['action_id'])
      exact!(action, member, order, arguments)
      raise Denied, 'idempotency_conflict' unless action.idempotency_key == key
      raise Denied, 'confirmation_expired' unless action.expires_at > @clock.call
      return action.result if %w[executed compensated].include?(action.status)
      raise Denied, 'stale_proposal' unless order.safe_action_state == action.before_state
      amount = arguments.fetch('amount_cents')
      unless order.refund_eligible && order.refund_key.nil? && amount <= order.refundable_balance_cents
        raise Denied, 'refund_ineligible'
      end
      before = order.safe_action_state
      order.update!(refunded_cents: order.refunded_cents + amount,
                    refund_key: key, status: 'refunded')
      result = { 'order_id' => order.reference, 'amount_cents' => amount, 'status' => 'refunded' }
      ActionReversalLog.create!(tool_action: action, organization_id: order.organization_id,
                               actor_id: member.user_id, event: 'execution',
                               before_state: before, after_state: order.safe_action_state)
      action.update!(status: 'executed', result: result)
      result
    end
    end
  rescue AiIncidentControl::HumanOnly
    raise Denied, 'human_only'
  end

  # Called only after the existing authoritative signed-approval policy succeeds.
  # Existing approvals remain compatible, but all mutations now share this ledger.
  def execute_approved(member, order, arguments, key)
    existing = ToolAction.find_by(organization_id: order.organization_id, idempotency_key: key)
    if existing
      exact!(existing, member, order, arguments)
      receipt = sign({ 'kind' => 'confirmation', 'organization_id' => existing.organization_id,
                       'actor_id' => existing.actor_id, 'order_id' => existing.order_id,
                       'arguments' => existing.arguments, 'action_id' => existing.id,
                       'exp' => existing.expires_at.to_i })
    else
      proposal = preview(member, order, arguments)
      receipt = confirm(member, order, proposal[:proposal_receipt], key)[:approval_receipt]
    end
    execute(member, order, arguments, key, receipt)
  end

  private

  def sign(claims)
    AgentApproval.safe_receipt(claims, verifier: @verifier)
  end

  def verify(receipt, kind, member, order)
    claims = AgentApproval.safe_claims(receipt, verifier: @verifier)
    unless claims.is_a?(Hash) && claims['kind'] == kind &&
           claims['organization_id'] == order.organization_id &&
           claims['actor_id'] == member.user_id && claims['order_id'] == order.id &&
           claims['exp'].is_a?(Integer) && claims['exp'] > @clock.call.to_i
      raise Denied, 'confirmation_required'
    end
    claims
  end

  def exact!(action, member, order, arguments)
    unless action.organization_id == order.organization_id && action.actor_id == member.user_id &&
           action.order_id == order.id && action.arguments == arguments
      raise Denied, 'idempotency_conflict'
    end
  end
end
