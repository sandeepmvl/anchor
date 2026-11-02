require 'json'

class AiBudgetCounter < ApplicationRecord
  DRIVERS = %w[uncached_input cached_input visible_output reasoning_output embedding_calls overhead].freeze
  METRICS = (DRIVERS + %w[input_tokens cached_tokens output_tokens reasoning_tokens embedding_tokens
    attempts cache_savings total]).freeze

  def self.configuration
    c = JSON.parse(File.read(Rails.root.join('config/ai_budgets.yml')))
    %w[price_version budget_version unit].each do |key|
      raise ArgumentError, 'invalid_cost_configuration' unless /\A[A-Za-z0-9_-]{1,80}\z/.match?(c.fetch(key))
    end
    raise ArgumentError, 'invalid_fixture_flag' unless [true, false].include?(c.fetch('fixture'))
    c.fetch('models').each_value do |rates|
      rates.each_value { |n| raise ArgumentError, 'invalid_price' unless n.is_a?(Integer) && n >= 0 }
    end
    c.fetch('budgets').each_value do |b|
      unless b.fetch('limit').is_a?(Integer) && b['limit'] > 0 &&
          b.fetch('warning_percent').is_a?(Integer) && (1..100).cover?(b['warning_percent'])
        raise ArgumentError, 'invalid_budget'
      end
    end
    c
  end

  def warning?(config = self.class.configuration)
    b = config.fetch('budgets').fetch(feature)
    total * 100 >= b.fetch('limit') * b.fetch('warning_percent')
  end

  def self.dashboard(tenant_id:, window:)
    where(tenant_id: tenant_id, window: window).order(:feature).to_a
  end
end
