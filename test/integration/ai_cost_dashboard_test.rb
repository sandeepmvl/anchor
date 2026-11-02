# Normal Rails discovery delegates to an isolated child, never another Rails app
# in the test runner's process. Direct execution uses the same child mode.
if ENV['ANCHOR_COST_STANDALONE'] != '1'
  require_relative '../test_helper'
  require 'open3'
  require 'rbconfig'
  class AiCostDashboardTest < ActiveSupport::TestCase
    def test_accounting_integration_in_isolated_process
      output, status = Open3.capture2e(
        {'ANCHOR_COST_STANDALONE' => '1', 'RAILS_ENV' => 'test'},
        RbConfig.ruby, File.expand_path(__FILE__))
      assert status.success?, output
    end
  end
else
  require 'rails'
  require 'action_controller/railtie'
  require 'active_record'
  require 'action_dispatch/testing/integration'
  require 'minitest/autorun'
  require 'json'
  require 'fileutils'

  ROOT = File.expand_path('../..', __dir__)
  # Define identity fixtures before Rails installs application autoloaders.
  # These constants exist only in this isolated verification process.
  class ApplicationRecord < ActiveRecord::Base
    self.abstract_class = true
  end
  class Organization < ApplicationRecord; end
  class Membership < ApplicationRecord
    validates :role, inclusion: { in: %w[admin agent end_customer] }
    validates :user_id, uniqueness: { scope: :organization_id }
  end

  class CostVerificationApp < Rails::Application
    config.root = ROOT
    config.eager_load = false
    config.secret_key_base = 'offline-cost-verifier-' * 8
    config.hosts.clear
    config.logger = Logger.new(File::NULL)
    config.autoload_paths = []
    config.autoload_once_paths = []
    config.eager_load_paths = []
    # Also exclude paths Rails derives from its application directory list.
    %w[app/models app/controllers app/helpers app/services app/jobs app/mailers].each do |path|
      config.paths[path] = []
    end
    config.paths['config/initializers'] = []
    config.paths['config/environments'] = []
    config.paths['config/routes.rb'] = []
    config.session_store :cookie_store, key: '_cost_verifier'
    config.action_controller.allow_forgery_protection = false
  end
  CostVerificationApp.initialize!
  FileUtils.mkdir_p(File.join(ROOT, 'tmp/cost_profile'))
  ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: ':memory:')
  ActiveRecord::Schema.define do
    create_table(:organizations) { |t| t.string :name }
    create_table(:memberships) do |t|
      t.integer :organization_id
      t.integer :user_id
      t.string :role
    end
  end
  require File.join(ROOT, 'db/migrate/20260101001000_create_ai_usage_and_budget_counters')
  CreateAiUsageAndBudgetCounters.new.migrate(:up)
  require File.join(ROOT, 'app/models/ai_budget_counter')
  require File.join(ROOT, 'app/models/ai_usage_event')
  require File.join(ROOT, 'app/controllers/api/ai_costs_controller')
  require File.join(ROOT, 'app/controllers/admin/ai_costs_controller')
  require File.join(ROOT, 'app/services/ai_telemetry')
  AiTelemetry.sink = ->(record) {}
  Admin::AiCostsController.prepend_view_path(File.join(ROOT, 'app/views'))
  class CostTestSessionsController < ActionController::Base
    def create
      session[:user_id] = params[:user_id].to_i
      head :ok
    end
  end
  # Verify the same two product route declarations without loading unrelated
  # application controllers, authentication initializers, or route side effects.
  CostVerificationApp.routes.draw do
    post '/api/ai_costs', to: 'api/ai_costs#create'
    get '/admin/organizations/:organization_id/ai_costs', to: 'admin/ai_costs#index'
    post '/__cost_test_session', to: 'cost_test_sessions#create'
  end

  class AiCostDashboardTest < ActionDispatch::IntegrationTest
    def app = CostVerificationApp

    def setup
      [AiUsageEvent, AiBudgetCounter, Membership, Organization].each(&:delete_all)
      Organization.create!(id: 1, name: 'one')
      Organization.create!(id: 2, name: 'two')
      Membership.create!(organization_id: 1, user_id: 10, role: 'admin')
      Membership.create!(organization_id: 1, user_id: 11, role: 'agent')
      Membership.create!(organization_id: 2, user_id: 20, role: 'admin')
      ENV['ANCHOR_COST_INGEST_SECRET'] = 'offline-service-secret'
    end

    def event
      {'tenant_id' => 1, 'attempt_id' => 'attempt1', 'logical_call_id' => 'call1',
       'attempt' => 1, 'feature' => 'chat', 'role' => 'base_completion', 'model' => 'standard',
       'window' => '2026-01', 'price_version' => 'fake-prices-v1',
       'budget_version' => 'feature-budgets-v1',
       'usage' => {'input' => 200, 'cached' => 50, 'output' => 50, 'reasoning' => 10}}
    end

    def deliver(e, secret: 'offline-service-secret')
      post '/api/ai_costs', params: JSON.generate(e), headers: {
        'CONTENT_TYPE' => 'application/json', 'X-Anchor-Cost-Secret' => secret}
    end

    def login(id)
      post '/__cost_test_session', params: {user_id: id}
      assert_response :success
    end

    def test_atomic_idempotent_delivery_and_conflict
      deliver(event)
      assert_response :success
      assert_equal true, JSON.parse(response.body)['inserted']
      before = AiBudgetCounter.first.attributes
      deliver(event)
      assert_equal false, JSON.parse(response.body)['inserted']
      assert_equal before, AiBudgetCounter.first.attributes
      assert_equal 1, AiUsageEvent.count
      changed = event
      changed['usage']['input'] = 201
      deliver(changed)
      assert_response :unprocessable_entity
      assert_equal before, AiBudgetCounter.first.attributes
      assert_equal 650, AiBudgetCounter.first.total
      assert_equal 50, AiBudgetCounter.first.cache_savings
      assert_raises(ActiveRecord::ReadOnlyRecord) { AiUsageEvent.first.update!(total: 0) }
    end

    def test_service_auth_and_content_rejection
      deliver(event, secret: 'wrong')
      assert_response :unauthorized
      deliver(event.merge('prompt' => 'must not be accepted'))
      assert_response :unprocessable_entity
      assert_equal 0, AiUsageEvent.count
      ENV.delete('ANCHOR_COST_INGEST_SECRET')
      deliver(event)
      assert_response :service_unavailable
    end

    def test_dashboard_requires_the_authorized_tenant_admin
      deliver(event)
      get '/admin/organizations/1/ai_costs', params: {window: '2026-01'}
      assert_response :unauthorized
      login(11)
      get '/admin/organizations/1/ai_costs', params: {window: '2026-01', role: 'admin'}
      assert_response :forbidden
      login(20)
      get '/admin/organizations/1/ai_costs', params: {window: '2026-01'}
      assert_response :forbidden
      get '/admin/organizations/2/ai_costs', params: {window: '2026-01'}
      assert_response :success
      refute_includes response.body, '<td>650</td>'
    end

    def test_exact_threshold_and_counter_only_dashboard
      deliver(event)
      counter = AiBudgetCounter.first
      counter.update!(total: 31999)
      refute counter.warning?
      counter.update!(total: 32000)
      assert counter.warning?
      login(10)
      queries = []
      guard = lambda do |*args|
        sql = args.last[:sql]
        queries << sql
        raise 'dashboard scanned the ledger' if /\bai_usage_events\b/i.match?(sql)
      end
      ActiveSupport::Notifications.subscribed(guard, 'sql.active_record') do
        get '/admin/organizations/1/ai_costs', params: {window: '2026-01'}
        assert_response :success
        assert_includes response.body, 'WARNING'
        assert AiBudgetCounter.dashboard(tenant_id: 1, window: '2026-01').first.warning?
      end
      assert queries.any? { |sql| sql.include?('ai_budget_counters') }
      invalid = Marshal.load(Marshal.dump(AiBudgetCounter.configuration))
      invalid['budgets']['chat']['warning_percent'] = 0
      path = Rails.root.join('config/ai_budgets.yml')
      original_read = File.method(:read)
      File.define_singleton_method(:read) do |name, *args|
        name.to_s == path.to_s ? JSON.generate(invalid) : original_read.call(name, *args)
      end
      assert_raises(ArgumentError) { AiBudgetCounter.configuration }
    ensure
      File.define_singleton_method(:read, original_read) if original_read
    end

    def test_thirty_ticket_dashboard_demo
      path = File.join(ROOT, 'tmp/cost_profile/events.json')
      unless File.exist?(path)
        require 'open3'
        output, status = Open3.capture2e('python3', 'ops/load/profile_tickets.py', '--costs', chdir: ROOT)
        assert status.success?, output
      end
      events = JSON.parse(File.read(path))
      assert_equal 120, events.length
      events.each do |e|
        deliver(e)
        assert_response :success
      end
      before = AiBudgetCounter.order(:feature).map(&:attributes)
      deliver(events.first)
      assert_equal false, JSON.parse(response.body)['inserted']
      assert_equal before, AiBudgetCounter.order(:feature).map(&:attributes)
      assert_equal [37650, 21900], AiBudgetCounter.order(:feature).pluck(:total)
      login(10)
      get '/admin/organizations/1/ai_costs', params: {window: '2026-01'}
      assert_response :success
      assert_includes response.body, '37650'
      assert_includes response.body, '21900'
      File.write(File.join(ROOT, 'tmp/cost_profile/dashboard.html'), response.body)
      puts 'Anchor: same ticket, auditable costs; tenant dashboard updated without conversation content.'
      puts '30 tickets: chat=37650 rag=21900; replay unchanged; chat warning only.'
      puts 'Rendered tenant-admin dashboard: tmp/cost_profile/dashboard.html'
    end
  end
end
