require 'minitest/autorun'
require 'active_record'
require 'active_support/all'
require 'tmpdir'

# Rails test discovery must not replace the application's database connection.
# Run this standalone service suite in a child process when Rails is booted.
if defined?(Rails) && Rails.respond_to?(:application) && Rails.application
  require 'rbconfig'
  class SafeToolActionsProcessTest < Minitest::Test
    def test_standalone_service_suite
      assert system(RbConfig.ruby, File.expand_path(__FILE__)),
             'standalone safe-action service tests failed'
    end
  end
else
# Standalone offline service integration; never touches Anchor's app databases.
unless defined?(Rails)
  module Rails
    def self.env = ActiveSupport::StringInquirer.new('test')
  end
end
SAFE_ACTION_TEST_DIRECTORY = Dir.mktmpdir('anchor-safe-actions')
Minitest.after_run { FileUtils.remove_entry(SAFE_ACTION_TEST_DIRECTORY) }
ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: File.join(SAFE_ACTION_TEST_DIRECTORY, 'test.sqlite3'))
ActiveRecord::Migration.verbose = false
ActiveRecord::Schema.define do
  create_table :organizations do |t|
    t.string :name
  end
  create_table :orders do |t|
    t.integer :organization_id, null: false
    t.string :reference
    t.string :payment_id
    t.string :status
    t.boolean :refund_eligible, default: true
    t.integer :refund_limit_cents, default: 10000
    t.integer :refunded_cents, default: 0
    t.string :refund_key
    t.timestamps
  end
end
require_relative '../../app/models/application_record'
class Organization < ApplicationRecord; end
require_relative '../../app/models/order'
require_relative '../../app/models/agent_approval'
require_relative '../../app/models/tool_action'
require_relative '../../app/models/action_reversal_log'
require_relative '../../db/migrate/20260101001300_create_tool_actions_and_reversal_logs'
CreateToolActionsAndReversalLogs.new.migrate(:up)
require_relative '../../app/services/tool_action_executor'
require_relative '../../app/services/tool_action_reverser'

class SafeToolActionsTest < Minitest::Test
  Member = Struct.new(:organization_id, :user_id, :role)

  def setup
    ActionReversalLog.delete_all
    ToolAction.delete_all
    Order.delete_all
    Organization.delete_all
    @org = Organization.create!(name: 'Anchor test tenant')
    @order = Order.create!(organization_id: @org.id, reference: 'A100', payment_id: 'fake_A100',
                           status: 'delivered', refund_limit_cents: 10000)
    @agent = Member.new(@org.id, 7, 'agent')
    @admin = Member.new(@org.id, 8, 'admin')
    @args = { 'order_id' => 'A100', 'amount_cents' => 2500 }
    @now = Time.utc(2026, 1, 1)
    @verifier = ActiveSupport::MessageVerifier.new('offline-test-key-' * 4, serializer: JSON)
    @executor = ToolActionExecutor.new(verifier: @verifier, local: true, clock: -> { @now })
    @reverser = ToolActionReverser.new(executor: @executor, local: true)
  end

  def confirm
    proposal = @executor.preview(@agent, @order, @args)
    @executor.confirm(@agent, @order, proposal[:proposal_receipt], 'refund_A100_1')
  end

  def execute(confirmation, arguments = @args, member = @agent)
    @executor.execute(member, @order, arguments, 'refund_A100_1', confirmation[:approval_receipt])
  end

  def test_preview_execute_retry_and_compensate
    before = @order.attributes
    preview = @executor.preview(@agent, @order, @args)
    assert_equal before, @order.reload.attributes
    assert_equal 0, ToolAction.count
    assert_equal 0, ActionReversalLog.count
    assert_equal 7500, preview[:after_balance_cents]
    confirmation = @executor.confirm(@agent, @order, preview[:proposal_receipt], 'refund_A100_1')
    result = execute(confirmation)
    assert_equal 7500, @order.reload.refundable_balance_cents
    assert_equal result, execute(confirmation)
    assert_equal 1, ActionReversalLog.where(event: 'execution').count
    assert_raises(ToolActionExecutor::Denied) { execute(confirmation, @args.merge('amount_cents' => 100)) }
    action = ToolAction.find(confirmation[:tool_action_id])
    assert_raises(ToolActionExecutor::Denied) { @reverser.reverse(@agent, action) }
    reversed = @reverser.reverse(@admin, action)
    assert_equal 10000, @order.reload.refundable_balance_cents
    assert_equal 'delivered', @order.status
    assert_nil @order.refund_key
    assert_equal reversed[:event_id], @reverser.reverse(@admin, action)[:event_id]
    assert_equal result, execute(confirmation)
    assert_equal 10000, @order.reload.refundable_balance_cents
    assert_equal 2, ActionReversalLog.count
    puts({ dry_run_balance: 10000, proposed_balance: 7500,
           executions: 1, changed_amount: 'rejected', restored_balance: 10000,
           compensation_events: 1, external_payment_provider_verified: false }.to_json)
  end

  def test_confirmation_binds_actor_tenant_resource_arguments_and_expiry
    confirmation = confirm
    assert_raises(ToolActionExecutor::Denied) { execute(confirmation, @args, @admin) }
    stranger = Member.new(@org.id + 1, @agent.user_id, 'agent')
    assert_raises(ToolActionExecutor::Denied) { execute(confirmation, @args, stranger) }
    other = Order.create!(organization_id: @org.id, reference: 'A101', payment_id: 'fake_A101', status: 'delivered')
    assert_raises(ToolActionExecutor::Denied) do
      @executor.execute(@agent, other, @args.merge('order_id' => 'A101'), 'refund_A100_1', confirmation[:approval_receipt])
    end
    @now += 301
    assert_raises(ToolActionExecutor::Denied) { execute(confirmation) }
    assert_equal 0, @order.reload.refunded_cents
  end

  def test_stale_state_and_blind_compensation_are_rejected
    proposal = @executor.preview(@agent, @order, @args)
    @order.update!(status: 'processing')
    @order.update!(status: 'delivered')
    assert_raises(ToolActionExecutor::Denied) do
      @executor.confirm(@agent, @order, proposal[:proposal_receipt], 'refund_A100_1')
    end
    confirmation = confirm
    @order.update!(payment_id: 'fake_replaced')
    assert_raises(ToolActionExecutor::Denied) { execute(confirmation) }
    @order.reload
    confirmation = @executor.confirm(@agent, @order,
      @executor.preview(@agent, @order, @args)[:proposal_receipt], 'refund_A100_2')
    @executor.execute(@agent, @order, @args, 'refund_A100_2', confirmation[:approval_receipt])
    @order.update!(refunded_cents: 3000)
    assert_raises(ToolActionExecutor::Denied) do
      @reverser.reverse(@admin, ToolAction.find(confirmation[:tool_action_id]))
    end
    assert_equal 3000, @order.reload.refunded_cents
  end

  def test_missing_confirmation_and_missing_production_adapter_fail_closed
    assert_raises(ToolActionExecutor::Denied) do
      @executor.execute(@agent, @order, @args, 'refund_A100_1', nil)
    end
    confirmation = confirm
    production = ToolActionExecutor.new(verifier: @verifier, local: false, clock: -> { @now })
    error = assert_raises(ToolActionExecutor::Denied) do
      production.execute(@agent, @order, @args, 'refund_A100_1', confirmation[:approval_receipt])
    end
    assert_equal 'payment_adapter_required', error.message
    assert_equal 0, ActionReversalLog.count
    assert_equal 10000, @order.reload.refundable_balance_cents
  end
end
end # Standalone suite branch; Rails discovery uses the child-process test.
