class CreateAiBudgetReservations < ActiveRecord::Migration[8.1]
  def change
    create_table :ai_budget_reservations do |t|
      t.string :attempt_id, null: false
      t.bigint :tenant_id, null: false
      t.string :actor, null: false
      t.string :feature, null: false
      t.string :day, null: false
      t.bigint :bound, null: false
      t.bigint :actual
      t.string :state, null: false, default: 'pending'
      t.timestamps
    end
    add_index :ai_budget_reservations, :attempt_id, unique: true
    add_index :ai_budget_reservations, [:tenant_id, :day]
    add_check_constraint :ai_budget_reservations,
      "bound > 0 AND (actual IS NULL OR (actual >= 0 AND actual <= bound))",
      name: 'valid_ai_reservation_amount'
    create_table :ai_quota_events do |t|
      t.bigint :tenant_id, null: false
      t.string :scope, null: false
      t.string :subject, null: false
      t.float :admitted_at, null: false
    end
    add_index :ai_quota_events, [:tenant_id, :scope, :subject, :admitted_at],
      name: 'ai_quota_event_lookup'
  end
end
