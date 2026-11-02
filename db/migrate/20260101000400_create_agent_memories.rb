class CreateAgentMemories < ActiveRecord::Migration[8.1]
  def change
    create_table :agent_memories do |t|
      t.integer :tenant_id, null: false
      t.integer :conversation_id, null: false
      t.string :run_key, null: false
      t.text :summary, null: false
      t.timestamps
    end
    add_index :agent_memories,
      [:tenant_id, :conversation_id, :run_key],
      unique: true, name: 'idx_agent_memory_run'
    add_index :agent_memories, [:tenant_id, :conversation_id]
  end
end
