class ToolActionReverser
  def initialize(executor: ToolActionExecutor.new,
                 local: Rails.env.development? || Rails.env.test?)
    @executor, @local = executor, local
  end

  def reverse(member, action)
    order = action.order
    @executor.authorize!(member, order, reverse: true)
    raise ToolActionExecutor::Denied, 'payment_adapter_required' unless @local
    ToolAction.transaction do
      order.lock!
      action.lock!
      existing = action.action_reversal_logs.find_by(event: 'compensation')
      return { status: 'compensated', event_id: existing.id } if existing
      unless action.status == 'executed'
        raise ToolActionExecutor::Denied, 'not_executed'
      end
      execution = action.action_reversal_logs.find_by!(event: 'execution')
      unless order.safe_action_state == execution.after_state
        raise ToolActionExecutor::Denied, 'reconciliation_required'
      end
      before = order.safe_action_state
      # Never rewind the version counter: compensation is another state change.
      order.update!(execution.before_state.except('lock_version'))
      event = ActionReversalLog.create!(
        tool_action: action, organization_id: order.organization_id,
        actor_id: member.user_id, event: 'compensation',
        before_state: before, after_state: order.safe_action_state)
      action.update!(status: 'compensated')
      { status: 'compensated', event_id: event.id,
        balance_cents: order.refundable_balance_cents }
    end
  end
end
