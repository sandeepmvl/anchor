require 'minitest/autorun'
require 'tmpdir'

if defined?(Rails) && Rails.respond_to?(:application) && Rails.application
  require 'open3'
  require 'rbconfig'

  # Rails discovers this file too. Keep the rehearsal models and database
  # connection in a child process so other project tests retain their models,
  # connection pools, schema, and transaction state.
  class AiIncidentPlaybookProcessTest < ActiveSupport::TestCase
    def test_isolated_incident_playbook_suite
      output, status = Open3.capture2e(RbConfig.ruby, File.expand_path(__FILE__),
        chdir: File.expand_path('../..', __dir__))
      assert status.success?, "Isolated incident playbook suite failed:\n#{output}"
    end
  end
else
  require_relative '../../labs/incident-response-playbooks/rehearsal'

# Standalone service integration test; intentionally does not boot Rails.
class AiIncidentPlaybookTest < Minitest::Test
  def setup
    @dir = Dir.mktmpdir('anchor-incident-')
    @path = File.join(@dir, 'rehearsal.sqlite3')
    IncidentRehearsal.connect(@path, create: true)
    @tenant = IncidentRehearsal::TENANT
    @now = Time.utc(2026, 1, 1, 12)
    @clock = -> { @now.to_f }
    @admin = Membership.find_by!(organization_id: @tenant, role: 'admin')
    signal = IncidentRehearsal.seed_spike(@path, clock: @clock)
    @event = AiIncidentControl.declare!(organization_id: @tenant, signal: signal, now: @now)
  end

  def teardown
    ActiveRecord::Base.connection_pool.disconnect!
    FileUtils.remove_entry(@dir)
  end

  def contain(member = @admin)
    AiIncidentControl.contain_incident!(member: member, organization_id: @tenant,
      incident_id: @event.fetch('incident_id'))
  end

  def checks
    { 'quality_pass' => true, 'budget_pass' => true, 'backlog_clear' => true,
      'uncertain_usage_reconciled' => true, 'observation_seconds' => 300 }
  end

  def test_cost_spike_records_severity_alert_and_original_deadline
    assert_equal 1, @event.fetch('severity')
    assert_equal %w[primary backup], @event.fetch('alert')
    assert_equal 10.0, @event.fetch('signal').fetch('ratio')
    assert_equal @now + 60, Time.iso8601(@event.fetch('human_banner_deadline'))
    refute AiIncidentControl.human_only?(@tenant), 'an alert is not authorized containment'
    quota = IncidentRehearsal.ledger(@path, clock: @clock)
    assert_equal 0, quota.incident_signal(tenant: IncidentRehearsal::OTHER, baseline: 1)['ratio']
    assert_raises(AiQuotaPolicy::IncidentWindow::Denied) do
      quota.incident_signal(tenant: @tenant, baseline: 0)
    end
  end

  def test_authorized_containment_returns_same_ticket_context_and_preserves_audit
    before = AiOperatingMode.find_by!(organization_id: @tenant).audit
    contain
    response = IncidentRehearsal.banner(@path)
    assert_equal 'human_only', response.fetch('status')
    assert_equal AiIncidentControl.config.fetch('message'), response.fetch('reply')
    IncidentRehearsal::CONTEXT.each do |key, value|
      assert_equal value, response.fetch('context').fetch(key)
    end
    audit = AiOperatingMode.find_by!(organization_id: @tenant).audit
    assert_equal before, audit.first(before.length)
    assert_equal @admin.user_id, audit.last.fetch('actor_id')
    assert_equal "incident:#{@event['incident_id']}:contain", audit.last.fetch('reason')
    assert audit.last.fetch('human_only')
    refute AiIncidentControl.human_only?(IncidentRehearsal::OTHER)
    deadline = Time.iso8601(@event.fetch('human_banner_deadline'))
    assert @now + 60 <= deadline
    refute @now + 60.001 <= deadline, 'a late banner must fail the SLA'
  end

  def test_non_admin_and_cross_tenant_admin_cannot_contain
    members = Membership.where(role: 'agent').to_a +
      Membership.where(organization_id: IncidentRehearsal::OTHER).to_a
    before = AiOperatingMode.find_by!(organization_id: @tenant).audit
    members.each { |member| assert_raises(AiIncidentControl::Unauthorized) { contain(member) } }
    assert_equal before, AiOperatingMode.find_by!(organization_id: @tenant).audit
    refute AiIncidentControl.human_only?(@tenant)
  end

  def test_missing_or_failed_recovery_checks_leave_human_only_active
    contain
    [{}, checks.merge('quality_pass' => false), checks.merge('budget_pass' => false),
      checks.merge('observation_seconds' => 299)].each do |incomplete|
      assert_raises(AiIncidentControl::ReleaseBlocked) do
        AiIncidentControl.recover_incident!(member: @admin, organization_id: @tenant,
          incident_id: @event['incident_id'], checks: incomplete, production: false)
      end
      assert AiIncidentControl.human_only?(@tenant)
    end
  end

  def test_production_recovery_fails_closed_without_real_evidence
    contain
    assert_raises(AiIncidentControl::ReleaseBlocked) do
      AiIncidentControl.recover_incident!(member: @admin, organization_id: @tenant,
        incident_id: @event['incident_id'], checks: checks, production: true, evidence: {})
    end
    assert AiIncidentControl.human_only?(@tenant)
  end

  def test_complete_offline_checks_allow_explicit_authorized_service_recovery
    contain
    AiIncidentControl.recover_incident!(member: @admin, organization_id: @tenant,
      incident_id: @event['incident_id'], checks: checks, production: false)
    refute AiIncidentControl.human_only?(@tenant)
    assert_equal 3, AiOperatingMode.find_by!(organization_id: @tenant).audit.length
  end
end
end # standalone child-process suite
