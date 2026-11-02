require 'json'

class AiUsageEvent < ApplicationRecord
  FIELDS = %w[tenant_id attempt_id logical_call_id attempt feature role model window
    price_version budget_version usage].freeze
  ROLES = %w[base_completion first_query_embedding extra_query_embedding rerank].freeze
  def readonly? = persisted?

  def self.nonnegative(n)
    raise ArgumentError, 'invalid_usage_integer' unless n.is_a?(Integer) && n >= 0
    n
  end

  def self.measure(event, config)
    e = event.to_h.deep_stringify_keys
    raise ArgumentError, 'unsafe_usage_fields' unless e.keys.sort == FIELDS.sort
    %w[attempt_id logical_call_id].each do |key|
      raise ArgumentError, 'invalid_attempt_identity' unless /\A[A-Za-z0-9_-]{1,80}\z/.match?(e[key].to_s)
    end
    unless e['tenant_id'].is_a?(Integer) && e['tenant_id'] > 0 &&
        e['attempt'].is_a?(Integer) && e['attempt'] > 0 && ROLES.include?(e['role']) &&
        /\A\d{4}-(0[1-9]|1[0-2])\z/.match?(e['window'].to_s)
      raise ArgumentError, 'invalid_usage_identity'
    end
    %w[price_version budget_version].each do |key|
      raise ArgumentError, 'unknown_cost_version' unless e[key] == config.fetch(key)
    end
    config.fetch('budgets').fetch(e['feature'])
    rates = config.fetch('models').fetch(e['model'])
    u = e.fetch('usage')
    raise ArgumentError, 'unsafe_usage_fields' unless u.is_a?(Hash) && u.keys.sort == %w[cached input output reasoning]
    i, c, o, r = %w[input cached output reasoning].map { |k| nonnegative(u[k]) }
    raise ArgumentError, 'invalid_usage_subsets' if c > i || r > o
    embedding = %w[first_query_embedding extra_query_embedding].include?(e['role'])
    metrics = AiBudgetCounter::METRICS.to_h { |k| [k, 0] }
    if embedding
      raise ArgumentError, 'invalid_embedding_usage' unless c.zero? && o.zero? && r.zero?
      total = i * rates.fetch('embedding')
      metrics['embedding_tokens'] = i
    else
      total = (i-c)*rates.fetch('input') + c*rates.fetch('cached') + o*rates.fetch('output')
      metrics.merge!('input_tokens' => i, 'cached_tokens' => c,
        'output_tokens' => o, 'reasoning_tokens' => r,
        'cache_savings' => c*(rates.fetch('input')-rates.fetch('cached')))
    end
    if e['attempt'] > 1 || %w[extra_query_embedding rerank].include?(e['role'])
      metrics['overhead'] = total
    elsif embedding
      metrics['embedding_calls'] = total
    else
      metrics.merge!('uncached_input' => (i-c)*rates.fetch('input'),
        'cached_input' => c*rates.fetch('cached'), 'visible_output' => (o-r)*rates.fetch('output'),
        'reasoning_output' => r*rates.fetch('output'))
    end
    metrics.merge!('total' => total, 'attempts' => 1)
    [e, metrics]
  end

  def self.ingest!(event)
    e, metrics = measure(event, AiBudgetCounter.configuration)
    # Canonical payload makes equivalent JSON key order a duplicate, not a conflict.
    canonical = e.sort.to_h
    canonical['usage'] = e.fetch('usage').sort.to_h
    payload = JSON.generate(canonical)
    transaction do
      Organization.find(e['tenant_id']).with_lock do
        existing = find_by(tenant_id: e['tenant_id'], attempt_id: e['attempt_id'])
        if existing
          raise ArgumentError, 'attempt_identity_conflict' unless existing.payload == payload
          return false
        end
        create!(e.slice('tenant_id', 'attempt_id', 'feature', 'window', 'price_version', 'budget_version')
          .merge('payload' => payload, 'total' => metrics['total']))
        counter = AiBudgetCounter.find_or_create_by!(e.slice('tenant_id', 'feature', 'window'))
        counter.lock!
        metrics.each { |key, value| counter[key] += value }
        counter.save!
      end
    end
    true
  end
end
