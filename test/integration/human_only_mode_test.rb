# Only the explicitly marked child may define the private schema and doubles.
# Rails discovery can load this file before Rails.application is initialized.
if ENV['ANCHOR_INCIDENT_HARNESS'] != '1'
  require 'minitest/autorun'
  require 'open3'
  require 'rbconfig'

  class HumanOnlyModeSubprocessTest < Minitest::Test
    def test_isolated_incident_harness
      output, status = Open3.capture2e(
        { 'ANCHOR_INCIDENT_HARNESS' => '1' },
        RbConfig.ruby, File.expand_path(__FILE__))
      puts output
      assert status.success?, "Incident harness failed:\n#{output}"
    end
  end
else
# Standalone product-seam integration test: ruby test/integration/human_only_mode_test.rb
require 'active_record'
require 'active_support/all'
require 'minitest/autorun'
require 'tmpdir'
require 'json'
require 'ostruct'

module Rails
  def self.env = ActiveSupport::StringInquirer.new('test')
  def self.logger = Logger.new(File::NULL)
end

INCIDENT_TEST_DIR = Dir.mktmpdir('anchor-incident-')
at_exit { FileUtils.remove_entry(INCIDENT_TEST_DIR) if File.exist?(INCIDENT_TEST_DIR) }
ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: File.join(INCIDENT_TEST_DIR, 'test.sqlite3'))
ActiveRecord::Schema.verbose = false
ActiveRecord::Schema.define do
  create_table(:organizations) { |t| t.string :name }
  create_table(:users) { |t| t.string :name }
  create_table(:memberships) do |t|
    t.integer :organization_id
    t.integer :user_id
    t.string :role
  end
  create_table(:ai_operating_modes) do |t|
    t.integer :organization_id, null: false
    t.boolean :human_only, default: false, null: false
    t.integer :revision, default: 0, null: false
    t.json :audit, default: [], null: false
  end
  add_index :ai_operating_modes, :organization_id, unique: true
  create_table(:orders) do |t|
    t.integer :organization_id
    t.string :reference
    t.integer :refunded_cents, default: 0
    t.integer :total_cents, default: 5000
    t.boolean :refund_eligible, default: true
    t.string :refund_key
    t.string :status, default: 'paid'
  end
  create_table(:tool_actions) do |t|
    t.integer :organization_id
    t.integer :actor_id
    t.integer :order_id
    t.string :idempotency_key
    t.json :arguments
    t.json :before_state
    t.json :result
    t.datetime :expires_at
    t.string :status, default: 'pending'
  end
  add_index :tool_actions, [:organization_id, :idempotency_key], unique: true
  create_table(:action_reversal_logs) do |t|
    t.integer :tool_action_id
    t.integer :organization_id
    t.integer :actor_id
    t.string :event
    t.json :before_state
    t.json :after_state
  end
end

class ApplicationRecord < ActiveRecord::Base
  self.abstract_class = true
end
class Organization < ApplicationRecord; end
class User < ApplicationRecord; end
class Membership < ApplicationRecord; end
class Order < ApplicationRecord
  def refundable_balance_cents = total_cents - refunded_cents
  def safe_action_state
    { 'refunded_cents' => refunded_cents, 'status' => status, 'refund_key' => refund_key }
  end
end
class ToolAction < ApplicationRecord
  belongs_to :order
end
class ActionReversalLog < ApplicationRecord
  belongs_to :tool_action
end
# Receipt adapter double; the executor still signs and verifies bound claims.
class AgentApproval
  def self.safe_receipt(claims, verifier:) = verifier.generate(claims)
  def self.safe_claims(receipt, verifier:) = verifier.verified(receipt)
end
class AgentOrchestrationClient
  class << self
    attr_accessor :calls
  end
  self.calls = 0
  def execute(*args, **options)
    self.class.calls += 1
    { 'status' => 'completed' }
  end
end

require_relative '../../app/models/ai_operating_mode'
require_relative '../../app/services/ai_incident_control'
require_relative '../../app/services/ai_task_dispatcher'
require_relative '../../app/services/tool_action_executor'

class HumanOnlyModeTest < Minitest::Test
  Task = Struct.new(:organization_id, :payload, :key, :result) do
    def run!(ingestion:, agent:)
      self.result = agent.call(payload, key)
    end
  end

  def setup
    [ActionReversalLog, ToolAction, Order, AiOperatingMode, Membership, User, Organization].each(&:delete_all)
    @org = Organization.create!(name: 'Acme')
    @other = Organization.create!(name: 'Other')
    admin_user = User.create!(name: 'Incident admin')
    agent_user = User.create!(name: 'Support agent')
    @admin = Membership.create!(organization_id: @org.id, user_id: admin_user.id, role: 'admin')
    @agent = Membership.create!(organization_id: @org.id, user_id: agent_user.id, role: 'agent')
    @order = Order.create!(organization_id: @org.id, reference: 'A17')
    @arguments = { 'order_id' => 'A17', 'amount_cents' => 2500 }
    @executor = ToolActionExecutor.new(local: true,
      verifier: ActiveSupport::MessageVerifier.new('incident-test-only', serializer: JSON))
    AgentOrchestrationClient.calls = 0
  end

  def change(value, member: @admin, organization_id: @org.id)
    AiIncidentControl.change!(member: member, organization_id: organization_id,
      human_only: value, reason: value ? 'Unsafe refund proposals' : 'Reviewed and reconciled')
  end

  def confirmation(key)
    proposal = @executor.preview(@agent, @order, @arguments)
    @executor.confirm(@agent, @order, proposal.fetch(:proposal_receipt), key).fetch(:approval_receipt)
  end

  def test_authorized_stop_blocks_previously_queued_task_with_full_context
    context = { 'message' => 'Refund A17', 'history' => ['Customer confirmed 2500 cents'],
      'evidence' => ['order:A17', 'refund-policy:v1'], 'proposed_action' => 'issue_refund',
      'order_id' => 'A17', 'amount_cents' => 2500, 'run_id' => 'refund-42',
      'attempts' => ['primary timeout'], 'conversation_id' => 17 }
    task = Task.new(@org.id, { 'request' => context,
      'execution_context' => 'old-signed-authorization', 'request_id' => 'r1' }, 'refund-42')
    assert_raises(AiIncidentControl::Unauthorized) { change(true, member: @agent) }
    assert_raises(AiIncidentControl::Unauthorized) { change(true, organization_id: @other.id) }
    change(true)
    AiTaskDispatcher.run(task)
    assert_equal 0, AgentOrchestrationClient.calls
    assert_equal 'human_only', task.result.fetch('status')
    assert_equal context.merge('idempotency_key' => 'refund-42'), task.result.fetch('context')
    refute_includes JSON.generate(task.result), 'old-signed-authorization'
    refute AiIncidentControl.human_only?(@other.id)
    assert_equal 1, AiOperatingMode.find_by!(organization_id: @org.id).audit.length
    puts 'Refund A17 queued before the kill switch: full-context human handoff; model calls=0.'
    change(false)
    AiTaskDispatcher.run(Task.new(@org.id, task.payload, task.key))
    assert_equal 1, AgentOrchestrationClient.calls
    assert_equal 2, AiOperatingMode.find_by!(organization_id: @org.id).revision
  end

  def test_stale_confirmation_and_authorization_cannot_execute_after_stop
    receipt = confirmation('refund-42')
    @executor.authorize!(@agent, @order) # Already passed before the incident.
    change(true)
    error = assert_raises(ToolActionExecutor::Denied) do
      @executor.execute(@agent, @order, @arguments, 'refund-42', receipt)
    end
    assert_equal 'human_only', error.message
    assert_equal 0, @order.reload.refunded_cents
    assert_equal 0, ActionReversalLog.count
    change(false)
    result = @executor.execute(@agent, @order, @arguments, 'refund-42', receipt)
    assert_equal result, @executor.execute(@agent, @order, @arguments, 'refund-42', receipt)
    assert_equal 2500, @order.reload.refunded_cents
    assert_equal 1, ActionReversalLog.count
    assert_raises(ToolActionExecutor::Denied) do
      @executor.execute(@agent, @order, @arguments.merge('amount_cents' => 3000), 'refund-42', receipt)
    end
    assert_equal 2500, @order.reload.refunded_cents
    puts 'Retried/fallback action identity refund-42: local refund executions=1.'
  end

  def test_stop_between_initial_authorization_and_locked_execution
    receipt = confirmation('refund-race')
    # Inject the switch change immediately after the real initial check.
    executor = @executor
    original = executor.method(:authorize!)
    admin, org = @admin, @org
    executor.define_singleton_method(:authorize!) do |member, order, **options|
      original.call(member, order, **options)
      AiIncidentControl.change!(member: admin, organization_id: org.id,
        human_only: true, reason: 'Race drill')
    end
    error = assert_raises(ToolActionExecutor::Denied) do
      executor.execute(@agent, @order, @arguments, 'refund-race', receipt)
    end
    assert_equal 'human_only', error.message
    assert_equal 0, @order.reload.refunded_cents
    assert_equal 0, ActionReversalLog.count
  end

  def test_missing_real_infrastructure_evidence_refuses_production_release
    assert_raises(AiIncidentControl::ReleaseBlocked) do
      AiIncidentControl.release_check!(production: true, evidence: {})
    end
    puts 'Offline smoke: live providers, HTTP/UI acceptance, multi-host revocation and remote payments NOT verified; production evidence gate refuses release.'
  end
end
end # Standalone harness; never define its doubles in the Rails process.
