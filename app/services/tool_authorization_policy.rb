require 'yaml'

class ToolAuthorizationPolicy
  class Denied < StandardError; end
  ACTIONS = %w[lookup_order check_refund issue_refund].freeze

  def self.settings
    YAML.safe_load_file(File.expand_path('../../config/agent_policy.yml', __dir__))
      .fetch('backend_authorization')
  end

  # Called only by trusted confirmation code after its existing approval gates.
  # This method is deliberately not an HTTP endpoint.
  def self.approval_receipt(membership:, order:, amount_cents:, idempotency_key:)
    Rails.application.message_verifier('tool-approval').generate(
      { 'membership_id' => membership.id, 'organization_id' => order.organization_id,
        'order_id' => order.id, 'amount_cents' => amount_cents,
        'idempotency_key' => idempotency_key },
      purpose: settings.fetch('approval_purpose'), expires_in: 60)
  end

  def initialize(membership)
    @membership = membership
  end

  def authorize!(action, arguments)
    raise Denied, 'action_denied' unless ACTIONS.include?(action)
    grants = self.class.settings.fetch('roles')
    raise Denied, 'role_denied' unless @membership.tool_action_allowed?(action, grants)
    keys = action == 'issue_refund' ? %w[order_id amount_cents] : %w[order_id]
    raise Denied, 'invalid_arguments' unless arguments.is_a?(Hash) && arguments.keys.sort == keys.sort
    reference = arguments['order_id']
    raise Denied, 'invalid_arguments' unless reference.is_a?(String) && reference.match?(/\A[A-Z][0-9]{3}\z/)
    if action == 'issue_refund'
      amount = arguments['amount_cents']
      raise Denied, 'invalid_arguments' unless amount.is_a?(Integer) && amount.between?(1, 1_000_000)
    end
    Order.where(organization_id: @membership.organization_id).lock.find_by!(reference: reference)
  rescue ActiveRecord::RecordNotFound
    raise Denied, 'resource_denied'
  end

  def approval!(receipt, order, amount, key)
    raise Denied, 'approval_required' unless receipt.is_a?(String) && key.is_a?(String) && key.match?(/\A[A-Za-z0-9_-]{1,200}\z/)
    expected = { 'membership_id' => @membership.id, 'organization_id' => order.organization_id,
                 'order_id' => order.id, 'amount_cents' => amount, 'idempotency_key' => key }
    actual = Rails.application.message_verifier('tool-approval').verified(
      receipt, purpose: self.class.settings.fetch('approval_purpose'))
    raise Denied, 'approval_required' unless actual == expected
  end
end
