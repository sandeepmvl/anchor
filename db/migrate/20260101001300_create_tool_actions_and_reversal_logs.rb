class CreateToolActionsAndReversalLogs < ActiveRecord::Migration[8.1]
  def change
    add_column :orders, :lock_version, :integer, null: false, default: 0
    create_table :tool_actions do |t|
      t.integer :organization_id, null: false
      t.integer :actor_id, null: false
      t.references :order, null: false, foreign_key: true
      t.string :idempotency_key, null: false
      t.json :arguments, null: false
      t.json :before_state, null: false
      t.json :result
      t.string :status, null: false, default: 'confirmed'
      t.datetime :expires_at, null: false
      t.timestamps
    end
    add_index :tool_actions, [:organization_id, :idempotency_key], unique: true
    create_table :action_reversal_logs do |t|
      t.references :tool_action, null: false, foreign_key: true
      t.integer :organization_id, null: false
      t.integer :actor_id, null: false
      t.string :event, null: false
      t.json :before_state, null: false
      t.json :after_state, null: false
      t.timestamps
    end
    add_index :action_reversal_logs, [:tool_action_id, :event], unique: true,
              name: 'unique_tool_action_audit_event'
  end
end
