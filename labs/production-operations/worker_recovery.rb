require 'sqlite3'
require 'json'
require 'yaml'
require 'fileutils'
require 'timeout'

module AnchorOperations
  ROOT = File.expand_path('../..', __dir__)
  LIMITATIONS = 'Offline SQLite fixture only: Solid Queue claims, external tools, deployed traces, paging and production recovery not verified.'

  def self.maintenance
    policy = YAML.safe_load(File.read(File.join(ROOT, 'ops/maintenance/schedule.yml')))
    required = %w[backup_verification index_maintenance credential_rotation quota_and_cost model_deprecation runbook_and_rotation]
    tasks = policy.fetch('tasks')
    raise 'missing maintenance tasks' unless (required - tasks.keys).empty?
    required.each do |name|
      %w[cadence owner evidence].each do |field|
        raise "missing #{name}.#{field}" if tasks.fetch(name)[field].to_s.strip.empty?
      end
    end
    { policy_valid: true, tasks: tasks, executed: false,
      note: 'Role ownership only. Record named acceptance, last evidence and next due date in the handoff.' }
  end

  class Drill
    def initialize(directory = File.join(ROOT, 'tmp/anchor-operations'))
      @directory = directory
      FileUtils.mkdir_p(@directory)
      @path = File.join(@directory, 'fixture.sqlite3')
    end

    def now
      Process.clock_gettime(Process::CLOCK_MONOTONIC)
    end

    def db
      connection = SQLite3::Database.new(@path)
      connection.results_as_hash = true
      connection.busy_timeout = 2000
      yield connection
    ensure
      connection&.close
    end

    def write(name, value)
      File.write(File.join(@directory, name), JSON.pretty_generate(value))
      value
    end

    def log(action, evidence)
      File.open(File.join(@directory, 'actions.jsonl'), 'a') do |file|
        file.puts JSON.generate(at: Time.now.utc.iso8601, actor: ENV.fetch('USER', 'exercise'), action: action, evidence: evidence)
      end
    end

    def setup
      raise 'attempt already exists; archive it before resetting' if File.exist?(@path)
      db do |c|
        c.execute_batch('CREATE TABLE tickets(id TEXT PRIMARY KEY, window TEXT, enqueued REAL, completed REAL, provider_ms REAL); CREATE TABLE actions(ticket_id TEXT PRIMARY KEY); CREATE TABLE meta(key TEXT PRIMARY KEY, value REAL);')
        enqueue(c, 'backlog')
      end
      sleep 2.05
      first = sample
      sleep 1.01
      second = sample
      alert = second[:pending] > 0 && second[:oldest_pending_ms] >= 2000 && second[:completed] == first[:completed]
      detected = now
      db { |c| c.execute('INSERT INTO meta VALUES (?, ?)', ['detected', detected]) }
      evidence = { alert: alert ? 'QueueAgeHigh' : nil, samples: [first, second],
        pending_latency_lower_bound_ms: second[:oldest_pending_ms], completed_p95_ms: nil,
        limitations: LIMITATIONS }
      write('alert.json', evidence)
      log('detected', evidence)
      evidence
    end

    def enqueue(c, window)
      timestamp = now
      20.times { |i| c.execute('INSERT INTO tickets(id, window, enqueued) VALUES (?, ?, ?)', ["#{window}-#{i}", window, timestamp]) }
    end

    def sample
      db do |c|
        pending = c.get_first_value('SELECT COUNT(*) FROM tickets WHERE completed IS NULL')
        oldest = c.get_first_value('SELECT MIN(enqueued) FROM tickets WHERE completed IS NULL')
        { pending: pending, completed: c.get_first_value('SELECT COUNT(*) FROM tickets WHERE completed IS NOT NULL'),
          oldest_pending_ms: oldest ? (now - oldest) * 1000 : 0 }
      end
    end

    def inspect_evidence
      started = now
      sleep 0.01
      { queue: sample, provider_span_ms: (now - started) * 1000,
        provider: 'deterministic offline fake; no network', limitations: LIMITATIONS }
    end

    def complete(id)
      # A local action ledger models an atomic tool boundary, not an external refund.
      started = now
      sleep 0.01
      duration = (now - started) * 1000
      db do |c|
        c.transaction do
          raise 'unknown ticket' unless c.get_first_value('SELECT id FROM tickets WHERE id = ?', [id])
          c.execute('INSERT OR IGNORE INTO actions(ticket_id) VALUES (?)', [id])
          c.execute('UPDATE tickets SET completed = ?, provider_ms = ? WHERE id = ? AND completed IS NULL', [now, duration, id])
        end
      end
    end

    def drain
      ids = db { |c| c.execute('SELECT id FROM tickets WHERE completed IS NULL ORDER BY id').map { |r| r['id'] } }
      ids.each { |id| complete(id) }
    end

    def restart
      alert = JSON.parse(File.read(File.join(@directory, 'alert.json')))
      raise 'no queue-stall evidence' unless alert['alert'] == 'QueueAgeHigh'
      reader, writer = IO.pipe
      pid = fork do
        reader.close
        stopping = false
        Signal.trap('TERM') { stopping = true }
        writer.write('ready')
        writer.close
        sleep 0.01 until stopping
        drain
        exit! 0
      end
      writer.close
      Timeout.timeout(5) do
        raise 'worker not ready' unless reader.read == 'ready'
        Process.kill('TERM', pid)
        _, status = Process.wait2(pid)
        pid = nil
        raise 'worker failed' unless status.success?
      end
      evidence = sample
      log('graceful_restart', evidence)
      evidence
    ensure
      reader&.close
      writer&.close unless writer&.closed?
      if pid
        Process.kill('KILL', pid) rescue Errno::ESRCH
        Process.wait(pid) rescue Errno::ECHILD
      end
    end

    def percentile(values)
      values.sort.fetch((values.length * 0.95).ceil - 1)
    end

    def counts
      db do |c|
        { tickets: c.get_first_value('SELECT COUNT(*) FROM tickets'),
          actions: c.get_first_value('SELECT COUNT(*) FROM actions') }
      end
    end

    def verify
      raise 'backlog has not drained' unless sample[:pending] == 0 && sample[:completed] == 20
      db { |c| enqueue(c, 'fresh') }
      drain
      rows = db { |c| c.execute('SELECT * FROM tickets') }
      fresh = rows.select { |r| r['window'] == 'fresh' }
      backlog = rows.select { |r| r['window'] == 'backlog' }
      latency = ->(r) { (r.fetch('completed') - r.fetch('enqueued')) * 1000 }
      elapsed = now - db { |c| c.get_first_value('SELECT value FROM meta WHERE key = ?', ['detected']) }
      report = { backlog_p95_ms: percentile(backlog.map(&latency)),
        fresh_p95_ms: percentile(fresh.map(&latency)), fresh_samples: fresh.length,
        fresh_window_start: fresh.map { |r| r['enqueued'] }.min,
        fresh_window_end: fresh.map { |r| r['completed'] }.max,
        clock: 'host monotonic seconds', elapsed_seconds: elapsed,
        counts: counts, pending: sample[:pending], limitations: LIMITATIONS }
      report[:passed] = report[:fresh_p95_ms] < 2000 && elapsed < 300 && report[:counts] == { tickets: 40, actions: 40 } && report[:pending] == 0
      write('recovery.json', report)
      log('verified', report)
      raise 'recovery objective failed; preserve evidence' unless report[:passed]
      report
    end

    def production_gate!
      raise 'Production recovery refused: deployed adapter, claim reconciliation and customer-delivery evidence missing; fixture evidence is insufficient.'
    end
  end
end

require 'time'
if $PROGRAM_NAME == __FILE__
  begin
    command = ARGV.fetch(0)
    drill = AnchorOperations::Drill.new
    result = case command
    when 'start' then drill.setup
    when 'inspect' then drill.inspect_evidence
    when 'restart' then drill.restart
    when 'verify' then drill.verify
    when 'maintenance' then AnchorOperations.maintenance
    when 'production-gate' then drill.production_gate!
    else raise 'expected start, inspect, restart, verify, maintenance or production-gate'
    end
    puts JSON.pretty_generate(result)
  rescue StandardError => error
    warn error.message
    exit 1
  end
end
