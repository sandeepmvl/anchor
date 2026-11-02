require 'minitest/autorun'
require 'json'
require_relative '../../app/services/inference_comparison_exporter'

class InferenceComparisonExporterTest < Minitest::Test
  Event = Struct.new(:attributes)
  class Store
    def initialize(rows)
      @rows = rows
    end
    def where(filters)
      self.class.new(@rows.select do |row|
        filters.all? do |key, value|
          actual = row.attributes[key.to_s]
          value.is_a?(Range) ? value.cover?(actual) : actual == value
        end
      end)
    end
    def each(&block)
      @rows.each(&block)
    end
  end
  class Members
    def find_by(user_id:, organization_id:, role:)
      user_id == 7 && organization_id == 1 && role == 'admin'
    end
  end

  def setup
    base = {'organization_id' => 1, 'created_at' => Time.utc(2026, 4, 2),
            'feature' => 'ticket_triage', 'input_tokens' => 150, 'output_tokens' => 20,
            'total' => 3, 'ticket_text' => 'private', 'customer_id' => 'secret'}
    @service = InferenceComparisonExporter.new(memberships: Members.new,
      events: Store.new([Event.new(base), Event.new(base.dup),
        Event.new(base.merge('organization_id' => 2, 'input_tokens' => 999999)),
        Event.new(base.merge('created_at' => Time.utc(2026, 5, 1)))]))
  end

  def export(**overrides)
    @service.export(**{user_id: 7, organization_id: 1, window: '2026-04', cost_unit: 'fixture_units'}.merge(overrides))
  end

  def test_only_selected_tenant_administrator_can_export
    [nil, 8, 9, 10].each do |user|
      assert_raises(InferenceComparisonExporter::Forbidden) { export(user_id: user) }
    end
    assert_raises(InferenceComparisonExporter::Forbidden) { export(organization_id: 2) }
    assert_raises(ArgumentError) { export(window: '2026-13') }
  end

  def test_daily_aggregation_has_no_content_or_other_tenants
    rows = export
    assert_equal 1, rows.length
    row = rows.first
    assert_equal 300, row['input_tokens']
    assert_equal 40, row['output_tokens']
    assert_equal 6, row['accounting_cost']
    assert_equal 'missing', row['latency']['provenance']
    assert_nil row['latency']['p99_ms']
    assert_equal %w[accounting_cost cost_unit date input_tokens latency output_tokens provenance workload], row.keys.sort
    refute_match(/private|secret|999999|customer_id|ticket_text/, JSON.generate(rows))
    puts JSON.generate(row)
  end

  def test_unknown_feature_is_not_exported_and_latency_has_provenance
    event = Event.new({'organization_id' => 1, 'created_at' => Time.utc(2026, 4, 2),
                       'feature' => 'customer@example.test', 'latency_ms' => 12})
    @service = InferenceComparisonExporter.new(memberships: Members.new, events: Store.new([event]))
    row = export.first
    assert_equal 'other', row['workload']
    assert_nil row['accounting_cost']
    assert_equal 12, row['latency']['p99_ms']
    assert_equal 'usage_events_warm_cold_unknown', row['latency']['provenance']
  end
end
