require 'date'
require 'time'

class InferenceComparisonExporter
  class Forbidden < StandardError; end
  WORKLOADS = %w[ticket_triage kb_embedding].freeze

  def initialize(memberships: Membership, events: AiUsageEvent)
    @memberships, @events = memberships, events
  end

  def export(user_id:, organization_id:, window:, cost_unit:)
    raise Forbidden unless user_id && @memberships.find_by(
      user_id: user_id, organization_id: organization_id, role: 'admin')
    raise ArgumentError, 'invalid month' unless /\A\d{4}-(0[1-9]|1[0-2])\z/.match?(window)
    start = Date.iso8601("#{window}-01")
    finish = start.next_month
    lower = Time.utc(start.year, start.month, 1)
    upper = Time.utc(finish.year, finish.month, 1)
    scoped = @events.where(organization_id: organization_id).where(created_at: lower...upper)
    groups = {}
    scoped.each do |event|
      a = event.attributes
      feature = a['feature'].to_s
      workload = WORKLOADS.include?(feature) ? feature : 'other'
      timestamp = a.fetch('created_at')
      timestamp = Time.parse(timestamp) if timestamp.is_a?(String)
      key = [timestamp.utc.to_date.iso8601, workload]
      bucket = groups[key] ||= {input_tokens: 0, output_tokens: 0, costs: [], latencies: [], cost_missing: false}
      bucket[:input_tokens] += Integer(a['input_tokens'] || 0)
      bucket[:input_tokens] += Integer(a['embedding_tokens'] || 0) if workload == 'kb_embedding'
      bucket[:output_tokens] += Integer(a['output_tokens'] || 0)
      cost = a['total']
      cost.nil? ? bucket[:cost_missing] = true : bucket[:costs] << Float(cost)
      latency = a['latency_ms']
      bucket[:latencies] << Float(latency) if latency && Float(latency).finite? && Float(latency) >= 0
    end
    groups.sort.map do |(day, workload), bucket|
      samples = bucket[:latencies].sort
      percentile = ->(fraction) { samples.empty? ? nil : samples[(samples.length * fraction).ceil - 1] }
      {'date' => day, 'workload' => workload,
       'input_tokens' => bucket[:input_tokens], 'output_tokens' => bucket[:output_tokens],
       'accounting_cost' => bucket[:cost_missing] ? nil : bucket[:costs].sum,
       'cost_unit' => cost_unit,
       'latency' => {'samples' => samples.length, 'p50_ms' => percentile.call(0.5),
                     'p99_ms' => percentile.call(0.99),
                     'provenance' => samples.empty? ? 'missing' : 'usage_events_warm_cold_unknown'},
       'provenance' => 'tenant_usage_aggregate'}
    end
  end
end
