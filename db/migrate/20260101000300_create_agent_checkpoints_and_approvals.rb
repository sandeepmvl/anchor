class CreateAgentCheckpointsAndApprovals < ActiveRecord::Migration[8.1]
  def change
    create_table :agent_checkpoints do |t|
      t.integer :tenant_id, null: false
      t.integer :conversation_id, null: false
      t.string :thread_id, null: false
      t.string :action_key, null: false
      t.string :status, null: false, default: "pending"
      t.text :context_json, null: false
      t.timestamps
    end
    add_index :agent_checkpoints, :thread_id, unique: true
    add_index :agent_checkpoints, [:tenant_id, :conversation_id]

    create_table :agent_approvals do |t|
      t.references :agent_checkpoint, null: false, foreign_key: true
      t.integer :tenant_id, null: false
      t.string :actor_id, null: false
      t.string :action_key, null: false
      t.string :nonce, null: false
      t.string :status, null: false, default: "approved"
      t.timestamps
    end
    add_index :agent_approvals, :agent_checkpoint_id,
              unique: true, name: "one_approval_per_checkpoint"
    add_index :agent_approvals, :nonce, unique: true
  end
end
