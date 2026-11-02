require 'sqlite3'
require 'json'
require 'date'
require 'securerandom'
require 'openssl'
require 'base64'

class AiQuotaPolicy
  class Denied < StandardError
    attr_reader :code, :status, :retry_after
    def initialize(code, status = 429, retry_after = nil)
      @code, @status, @retry_after = code, status, retry_after
      super(code)
    end
  end

  def self.default
    new(path: ActiveRecord::Base.connection_db_config.database,
      config: JSON.parse(Rails.root.join('config/ai_budgets.yml').read))
  end

  def self.release_check!(production:, config:, evidence: {})
    return unless production
    required = %w[quota_all_paid_paths quota_real_bounds quota_legacy_reconciled quota_shared_authority]
    unless config['fixture'] == false && required.all? { |key| evidence[key] == '1' }
      raise Denied.new('quota_release_evidence_missing', 503)
    end
  end

  # The application issues this context AFTER authenticating its customer.
  # It is not a substitute for customer authentication or ticket authorization.
  def self.issue_context(identity, secret:, now: Time.now.to_i)
    raise Denied.new('quota_identity_required', 401) unless secret.to_s.bytesize >= 32
    payload = Base64.urlsafe_encode64(JSON.generate(identity.merge('expires' => now + 300)), padding: false)
    payload + '.' + OpenSSL::HMAC.hexdigest('SHA256', secret, payload)
  end

  def self.verify_context(token, secret:, now: Time.now.to_i)
    raise Denied.new('quota_identity_required', 401) unless secret.to_s.bytesize >= 32
    payload, signature = token.to_s.split('.', 2)
    expected = OpenSSL::HMAC.hexdigest('SHA256', secret, payload.to_s)
    unless signature && signature.bytesize == expected.bytesize &&
        OpenSSL.fixed_length_secure_compare(signature, expected)
      raise Denied.new('quota_identity_required', 401)
    end
    claims = JSON.parse(Base64.urlsafe_decode64(payload))
    unless claims['expires'].is_a?(Integer) && claims['expires'] > now
      raise Denied.new('quota_identity_required', 401)
    end
    claims
  rescue JSON::ParserError, ArgumentError
    raise Denied.new('quota_identity_required', 401)
  end

  def initialize(path:, config:, clock: -> { Time.now.to_f })
    @path, @config, @clock = path.to_s, config, clock
    q = @config.fetch('quotas')
    positive!(q.fetch('daily_limit'))
    [q.fetch('tenant'), q.fetch('user'), *q.fetch('features').values].each do |rule|
      positive!(rule.fetch('limit')); positive!(rule.fetch('window_seconds'))
    end
    @config.fetch('budgets').each_value { |budget| positive!(budget.fetch('limit')) }
  end

  def positive!(value)
    raise Denied.new('quota_configuration_invalid', 503) unless value.is_a?(Integer) && value.positive?
    value
  end

  def identity!(identity)
    unless identity.is_a?(Hash) && identity['tenant_id'].is_a?(Integer) &&
        identity['tenant_id'].positive? && identity['actor'].is_a?(String) &&
        identity['actor'].match?(/\A[A-Za-z0-9_:@.-]{1,120}\z/) &&
        @config.fetch('quotas').fetch('features').key?(identity['feature'])
      raise Denied.new('quota_identity_required', 401)
    end
    identity
  end

  def transaction
    db = SQLite3::Database.new(@path)
    db.results_as_hash = true
    db.busy_timeout = 1500
    db.execute('BEGIN IMMEDIATE')
    begin
      value = yield db
      db.execute('COMMIT')
      value
    rescue Exception
      db.execute('ROLLBACK') if db.transaction_active?
      raise
    ensure
      db.close
    end
  rescue SQLite3::Exception
    raise Denied.new('quota_authority_unavailable', 503)
  end

  def rate!(identity)
    identity!(identity)
    transaction { |db| rate_in_transaction!(db, identity, @clock.call) }
    true
  end

  def rate_in_transaction!(db, identity, now)
    q = @config.fetch('quotas')
    rules = [['tenant', '*', q.fetch('tenant')],
      ['user', identity.fetch('actor'), q.fetch('user')],
      ['feature', identity.fetch('feature'), q.fetch('features').fetch(identity.fetch('feature'))]]
    waits = []
    rules.each do |scope, subject, rule|
      values = [identity.fetch('tenant_id'), scope, subject]
      db.execute('DELETE FROM ai_quota_events WHERE tenant_id=? AND scope=? AND subject=? AND admitted_at<=?',
        values + [now - rule.fetch('window_seconds')])
      rows = db.execute('SELECT admitted_at FROM ai_quota_events WHERE tenant_id=? AND scope=? AND subject=? ORDER BY admitted_at', values)
      if rows.length >= rule.fetch('limit')
        waits << [1, (rows[rows.length - rule.fetch('limit')]['admitted_at'] + rule.fetch('window_seconds') - now).ceil].max
      end
    end
    raise Denied.new('rate_limit', 429, waits.max) unless waits.empty?
    rules.each do |scope, subject, _rule|
      db.execute('INSERT INTO ai_quota_events(tenant_id,scope,subject,admitted_at) VALUES(?,?,?,?)',
        [identity.fetch('tenant_id'), scope, subject, now])
    end
  end

  def totals(db, tenant, day, feature = nil)
    sql = 'SELECT COALESCE(SUM(CASE WHEN state=\'settled\' THEN actual ELSE 0 END),0) AS spent, COALESCE(SUM(CASE WHEN state=\'pending\' THEN bound ELSE 0 END),0) AS reserved FROM ai_budget_reservations WHERE tenant_id=? AND day=?'
    args = [tenant, day]
    if feature
      sql += ' AND feature=?'; args << feature
    end
    db.get_first_row(sql, args).slice('spent', 'reserved')
  end

  def reserve!(identity, bound:, attempt_id:, rate: true)
    identity!(identity); positive!(bound)
    unless attempt_id.is_a?(String) && attempt_id.match?(/\A[A-Za-z0-9_-]{1,80}\z/)
      raise Denied.new('quota_attempt_invalid', 422)
    end
    now = @clock.call
    day = Time.at(now).utc.strftime('%Y-%m-%d')
    transaction do |db|
      if db.get_first_value('SELECT id FROM ai_budget_reservations WHERE attempt_id=?', [attempt_id])
        raise Denied.new('quota_attempt_reused', 409)
      end
      rate_in_transaction!(db, identity, now) if rate
      tenant = totals(db, identity.fetch('tenant_id'), day)
      feature = totals(db, identity.fetch('tenant_id'), day, identity.fetch('feature'))
      tenant_cap = @config.fetch('quotas').fetch('daily_limit')
      feature_cap = @config.fetch('budgets').fetch(identity.fetch('feature')).fetch('limit')
      if tenant.values.sum + bound > tenant_cap || feature.values.sum + bound > feature_cap
        retry_at = Time.utc(*Time.at(now).utc.to_date.next_day.to_s.split('-').map(&:to_i)).to_f
        raise Denied.new('budget_exhausted', 429, [1, (retry_at - now).ceil].max)
      end
      stamp = Time.at(now).utc.strftime('%Y-%m-%d %H:%M:%S')
      db.execute('INSERT INTO ai_budget_reservations(attempt_id,tenant_id,actor,feature,day,bound,state,created_at,updated_at) VALUES(?,?,?,?,?,?,\'pending\',?,?)',
        [attempt_id, identity.fetch('tenant_id'), identity.fetch('actor'), identity.fetch('feature'), day, bound, stamp, stamp])
      { 'reservation_id' => attempt_id, 'bound' => bound, 'day' => day }
    end
  end

  def settle!(identity, reservation_id:, actual:)
    identity!(identity)
    transaction do |db|
      row = db.get_first_row('SELECT * FROM ai_budget_reservations WHERE attempt_id=? AND tenant_id=? AND actor=? AND feature=?',
        [reservation_id, identity.fetch('tenant_id'), identity.fetch('actor'), identity.fetch('feature')])
      raise Denied.new('quota_reservation_unknown', 404) unless row
      unless actual.is_a?(Integer) && actual.between?(0, row['bound'])
        raise Denied.new('quota_bound_exceeded', 409)
      end
      if row['state'] == 'settled'
        raise Denied.new('quota_settlement_conflict', 409) unless row['actual'] == actual
      else
        db.execute('UPDATE ai_budget_reservations SET actual=?,state=\'settled\',updated_at=? WHERE id=?',
          [actual, Time.now.utc.strftime('%Y-%m-%d %H:%M:%S'), row['id']])
      end
      { 'reservation_id' => reservation_id, 'state' => 'settled', 'actual' => actual }
    end
  end

  def snapshot(identity)
    identity!(identity)
    day = Time.at(@clock.call).utc.strftime('%Y-%m-%d')
    transaction do |db|
      totals(db, identity.fetch('tenant_id'), day).merge('tenant_id' => identity.fetch('tenant_id'),
        'day' => day, 'cap' => @config.fetch('quotas').fetch('daily_limit'),
        'unit' => @config.fetch('unit'))
    end
  end
end
