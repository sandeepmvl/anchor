require 'minitest/autorun'
require_relative '../../labs/quotas-rate-limits/quota_burst_test'

class AiQuotaEnforcementTest < Minitest::Test
  def setup
    @directory = Dir.mktmpdir('anchor-quota-test-')
    @path = File.join(@directory, 'application.sqlite3')
    QuotaLab.prepare(@path)
    @now = Time.now.to_f
    @config = QuotaLab.config(rate: 3, cap: 100)
    @identity = QuotaLab::IDENTITY
    @policy = policy
  end

  def teardown
    FileUtils.remove_entry(@directory)
  end

  def policy
    AiQuotaPolicy.new(path: @path, config: @config, clock: -> { @now })
  end

  def test_concurrent_requests_share_tenant_window
    @config['quotas']['user']['limit'] = 100
    @config['quotas']['features']['chat']['limit'] = 100
    admitted = QuotaLab.concurrent(@path, @config)
    assert_equal 3, admitted
    assert_raises(AiQuotaPolicy::Denied) do
      policy.rate!(@identity.merge('actor' => 'another-user'))
    end
  end

  def test_limited_requests_return_429_and_retry_after
    3.times { assert_equal 200, QuotaLab.request(@policy).first }
    status, headers, body = QuotaLab.request(policy)
    assert_equal 429, status
    assert_equal 'rate_limit', body.fetch('error')
    assert_operator headers.fetch('retry-after').to_i, :>, 0
    @now += 61
    assert_equal 200, QuotaLab.request(policy).first
  end

  def test_user_and_feature_windows_are_independent
    @config['quotas']['tenant']['limit'] = 100
    @config['quotas']['user']['limit'] = 1
    @config['quotas']['features']['chat']['limit'] = 2
    policy.rate!(@identity)
    assert_raises(AiQuotaPolicy::Denied) { policy.rate!(@identity) }
    policy.rate!(@identity.merge('actor' => 'second'))
    assert_raises(AiQuotaPolicy::Denied) { policy.rate!(@identity.merge('actor' => 'third')) }
    policy.rate!(@identity.merge('actor' => 'third', 'feature' => 'rag'))
  end

  def test_reservations_survive_new_process_and_settlement_is_idempotent
    @config['quotas']['daily_limit'] = 5
    reservation = policy.reserve!(@identity, bound: 3, attempt_id: 'first')
    read, write = IO.pipe
    pid = fork do
      read.close
      write.write(JSON.generate(policy.snapshot(@identity)))
      write.close
      exit! 0
    end
    write.close
    snapshot = JSON.parse(read.read); read.close
    Process.wait(pid)
    assert $?.success?
    assert_equal 3, snapshot.fetch('reserved')
    assert_raises(AiQuotaPolicy::Denied) do
      policy.reserve!(@identity, bound: 3, attempt_id: 'second')
    end
    2.times { policy.settle!(@identity, reservation_id: reservation.fetch('reservation_id'), actual: 2) }
    assert_equal 2, policy.snapshot(@identity).fetch('spent')
    assert_equal 0, policy.snapshot(@identity).fetch('reserved')
    assert_raises(AiQuotaPolicy::Denied) { policy.settle!(@identity, reservation_id: 'first', actual: 1) }
    assert_raises(AiQuotaPolicy::Denied) { policy.reserve!(@identity, bound: 1, attempt_id: 'first') }
  end

  def test_tenant_isolation_and_unknown_outcomes
    @config['quotas']['daily_limit'] = 3
    policy.reserve!(@identity, bound: 3, attempt_id: 'uncertain')
    @now += 120
    assert_equal 3, policy.snapshot(@identity).fetch('reserved')
    assert_raises(AiQuotaPolicy::Denied) { policy.reserve!(@identity, bound: 1, attempt_id: 'blocked') }
    other = @identity.merge('tenant_id' => 91002)
    assert policy.reserve!(other, bound: 3, attempt_id: 'other')
  end

  def test_missing_database_and_missing_release_evidence_fail_closed
    unavailable = AiQuotaPolicy.new(path: File.join(@directory, 'missing', 'db'), config: @config)
    error = assert_raises(AiQuotaPolicy::Denied) { unavailable.rate!(@identity) }
    assert_equal 503, error.status
    assert_raises(AiQuotaPolicy::Denied) do
      AiQuotaPolicy.release_check!(production: true, config: @config, evidence: {})
    end
  end

  def test_unverified_context_does_not_dispatch
    QuotaLab::ChatProbeController.provider_calls = 0
    status, _, _ = QuotaLab.request(@policy, token: 'forged.context')
    assert_equal 401, status
    assert_equal 0, QuotaLab::ChatProbeController.provider_calls
  end

  def test_prepare_restores_application_database_after_success_and_failure
    original = begin
      ActiveRecord::Base.connection_db_config
    rescue ActiveRecord::ConnectionNotEstablished
      nil
    end
    application_path = File.join(@directory, 'original.sqlite3')
    quota_path = File.join(@directory, 'isolated-quota.sqlite3')
    begin
      ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: application_path)
      ActiveRecord::Base.connection.execute('CREATE TABLE application_sentinel (value INTEGER)')
      ActiveRecord::Base.connection.execute('INSERT INTO application_sentinel VALUES (42)')
      # A configured but disconnected pool must also be restored.
      ActiveRecord::Base.connection_pool.disconnect!
      QuotaLab.prepare(quota_path)
      assert_equal application_path, ActiveRecord::Base.connection_db_config.database
      assert_equal 42, ActiveRecord::Base.connection.select_value('SELECT value FROM application_sentinel')
      refute ActiveRecord::Base.connection.table_exists?(:ai_budget_reservations)
      # Migrating the same temporary database again fails on existing tables.
      assert_raises(StandardError) { QuotaLab.prepare(quota_path) }
      assert_equal application_path, ActiveRecord::Base.connection_db_config.database
      assert_equal 42, ActiveRecord::Base.connection.select_value('SELECT value FROM application_sentinel')
      puts 'Quota database preparation preserves the application database after success and failure.'
    ensure
      ActiveRecord::Base.remove_connection
      ActiveRecord::Base.establish_connection(original) if original
    end
  end
end
