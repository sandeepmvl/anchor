require 'active_record'
require 'action_controller/railtie'
require 'rack/mock'
require 'tmpdir'
require 'json'
require_relative '../../app/services/ai_quota_policy'
require_relative '../../app/controllers/concerns/ai_quota_enforcement'
require_relative '../../db/migrate/20260101001100_create_ai_budget_reservations'

module QuotaLab
  IDENTITY = { 'tenant_id' => 91001, 'actor' => 'customer-1', 'feature' => 'chat' }.freeze
  SECRET = 'offline-quota-context-secret-32-characters'.freeze

  def self.config(rate: 5, cap: 100)
    c = JSON.parse(File.read('config/ai_budgets.yml'))
    c['quotas']['daily_limit'] = cap
    %w[tenant user].each { |key| c['quotas'][key]['limit'] = rate }
    c['quotas']['features']['chat']['limit'] = rate
    c['budgets']['chat']['limit'] = cap
    c
  end

  def self.prepare(path)
    # Preserve configuration even when the original pool has no open connection.
    original = begin
      ActiveRecord::Base.connection_db_config
    rescue ActiveRecord::ConnectionNotEstablished
      nil
    end
    begin
      ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: path)
      ActiveRecord::Migration.verbose = false
      CreateAiBudgetReservations.new.migrate(:up)
    ensure
      # Remove the temporary pool before restoring the application's database.
      ActiveRecord::Base.remove_connection
      ActiveRecord::Base.establish_connection(original) if original
    end
  end

  class ChatProbeController < ActionController::API
    include AiQuotaEnforcement
    class_attribute :authority, :provider_calls, default: nil

    def quota_policy
      self.class.authority
    end

    def create
      # This is a bounded offline adapter, not a production pricing estimate.
      reservation = quota_policy.reserve!(quota_identity, bound: 3,
        attempt_id: SecureRandom.hex(16), rate: false)
      self.class.provider_calls += 1
      quota_policy.settle!(quota_identity, reservation_id: reservation.fetch('reservation_id'), actual: 2)
      render json: { reply: 'Offline refund-policy answer', provider: 'fake' }
    rescue AiQuotaPolicy::Denied => error
      response.set_header('Retry-After', error.retry_after.to_s) if error.retry_after
      render json: { error: error.code }, status: error.status
    end
  end

  def self.request(policy, identity: IDENTITY, token: nil)
    ENV['ANCHOR_QUOTA_SECRET'] = SECRET
    token ||= AiQuotaPolicy.issue_context(identity, secret: SECRET)
    ChatProbeController.authority = policy
    ChatProbeController.provider_calls ||= 0
    env = Rack::MockRequest.env_for('/chat', method: 'POST', input: '{"message":"refund policy?"}',
      'CONTENT_TYPE' => 'application/json', 'HTTP_X_ANCHOR_QUOTA_CONTEXT' => token)
    status, headers, body = ChatProbeController.action(:create).call(env)
    text = +''
    begin
      body.each { |part| text << part }
    ensure
      body.close if body.respond_to?(:close)
    end
    [status, headers, JSON.parse(text)]
  end

  def self.assert(value, message)
    raise message unless value
  end

  def self.concurrent(path, config, count: 12)
    # Two actual processes open independent connections to one authority.
    pipes = 2.times.map do
      read, write = IO.pipe
      pid = fork do
        read.close
        policy = AiQuotaPolicy.new(path: path, config: config)
        admitted = 0
        count.times do
          begin
            policy.reserve!(IDENTITY, bound: 3, attempt_id: SecureRandom.hex(16))
            admitted += 1
          rescue AiQuotaPolicy::Denied => error
            raise unless %w[rate_limit budget_exhausted].include?(error.code)
          end
        end
        write.write(admitted.to_s); write.close
        exit! 0
      end
      write.close
      [pid, read]
    end
    pipes.sum do |pid, read|
      value = read.read.to_i; read.close
      Process.wait(pid)
      assert($?.success?, 'concurrent worker failed')
      value
    end
  end

  def self.demo
    Dir.mktmpdir('anchor-quota-') do |dir|
      path = File.join(dir, 'application.sqlite3')
      prepare(path)
      now = 1_893_456_000.0
      c = config(rate: 5, cap: 100)
      policy = AiQuotaPolicy.new(path: path, config: c, clock: -> { now })
      ChatProbeController.provider_calls = 0
      burst = Array.new(50) { request(policy) }
      counts = burst.group_by { |status, _, body| status == 200 ? 'answered' : body.fetch('error') }.transform_values(&:length)
      assert(counts == { 'answered' => 5, 'rate_limit' => 45 }, counts.inspect)
      assert(burst.select { |s, _, _| s == 429 }.all? { |_, h, _| h['retry-after'].to_i.positive? }, 'Retry-After missing')
      puts JSON.generate(burst: counts, live_budget: policy.snapshot(IDENTITY))
      now += 61
      c['quotas']['daily_limit'] = 12
      c['budgets']['chat']['limit'] = 12
      %w[tenant user].each { |key| c['quotas'][key]['limit'] = 100 }
      c['quotas']['features']['chat']['limit'] = 100
      policy = AiQuotaPolicy.new(path: path, config: c, clock: -> { now })
      calls = ChatProbeController.provider_calls
      rejected = request(policy)
      assert(rejected[0] == 429 && rejected[2]['error'] == 'budget_exhausted', 'budget gate failed')
      assert(ChatProbeController.provider_calls == calls, 'denied request invoked provider')
      puts JSON.generate(after_rate_raise: rejected[2], provider_calls: calls,
        live_budget: policy.snapshot(IDENTITY))
      parallel_path = File.join(dir, 'parallel.sqlite3')
      prepare(parallel_path)
      parallel_config = config(rate: 100, cap: 9)
      admitted = concurrent(parallel_path, parallel_config)
      assert(admitted == 3, 'concurrent overspend')
      puts JSON.generate(two_process_reservations: admitted, maximum_reserved: 9)
      puts 'Offline smoke passed. Real customer authentication, prompt-chain wiring, provider pricing, invoice reconciliation, and multi-host authority are not verified; production release remains blocked.'
    end
  end
end

if $PROGRAM_NAME == __FILE__
  if ARGV == ['--rpc']
    begin
      input = JSON.parse($stdin.read)
      policy = AiQuotaPolicy.new(path: input.fetch('database'), config: input.fetch('config'), clock: -> { input.fetch('clock') })
      identity = input.fetch('identity')
      result = case input.fetch('operation')
      when 'reserve' then policy.reserve!(identity, bound: input.fetch('bound'), attempt_id: input.fetch('attempt_id'))
      when 'settle' then policy.settle!(identity, reservation_id: input.fetch('reservation_id'), actual: input.fetch('actual'))
      when 'snapshot' then policy.snapshot(identity)
      else raise 'unknown operation'
      end
      puts JSON.generate(result)
    rescue AiQuotaPolicy::Denied => error
      puts JSON.generate(error: error.code, status: error.status, retry_after: error.retry_after)
    end
  else
    QuotaLab.demo
  end
end
