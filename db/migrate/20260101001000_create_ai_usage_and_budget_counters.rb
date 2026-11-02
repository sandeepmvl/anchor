class CreateAiUsageAndBudgetCounters < ActiveRecord::Migration[8.1]
  def change
    create_table :ai_usage_events do |t|
      t.bigint :tenant_id, null: false
      t.string :attempt_id, null: false
      t.string :feature, null: false
      t.string :window, null: false
      t.string :price_version, null: false
      t.string :budget_version, null: false
      t.text :payload, null: false
      t.bigint :total, null: false
      t.timestamps
    end
    add_index :ai_usage_events, [:tenant_id, :attempt_id], unique: true
    create_table :ai_budget_counters do |t|
      t.bigint :tenant_id, null: false
      t.string :feature, null: false
      t.string :window, null: false
      %i[uncached_input cached_input visible_output reasoning_output embedding_calls overhead
        input_tokens cached_tokens output_tokens reasoning_tokens embedding_tokens attempts
        cache_savings total].each { |name| t.bigint name, null: false, default: 0 }
      t.timestamps
    end
    add_index :ai_budget_counters, [:tenant_id, :feature, :window], unique: true,
      name: 'ai_budget_counter_identity'
  end
end
