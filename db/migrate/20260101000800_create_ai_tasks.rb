class CreateAiTasks < ActiveRecord::Migration[8.1]
  def change
    create_table :ai_tasks do |t|
      t.bigint :organization_id, null: false
      t.string :submission_key, null: false
      t.string :payload_digest, null: false
      t.string :operation, null: false
      t.text :payload_json, null: false
      t.string :state, null: false, default: "queued"
      t.integer :progress, null: false, default: 0
      t.integer :total, null: false
      t.integer :attempts, null: false, default: 0
      t.datetime :available_at, null: false
      t.datetime :lease_until
      t.string :claim_token
      t.string :error_code
      t.timestamps
    end
    add_index :ai_tasks, [:organization_id, :submission_key], unique: true
    add_index :ai_tasks, [:state, :available_at]
    add_index :ai_tasks, [:state, :lease_until]
  end
end
