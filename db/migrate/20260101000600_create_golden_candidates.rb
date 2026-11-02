class CreateGoldenCandidates < ActiveRecord::Migration[8.1]
  def change
    create_table :answer_messages do |t|
      t.string :tenant_id, null: false
      t.string :customer_id, null: false
      t.string :conversation_id, null: false
      t.string :prompt_version, null: false
      t.json :retrieved_chunk_ids, null: false
      t.json :payload, null: false
      t.timestamps
    end
    add_index :answer_messages, [:tenant_id, :customer_id, :conversation_id], name: 'answer_owner'
    create_table :feedback_events do |t|
      t.references :answer_message, null: false, foreign_key: true, index: {unique: true}
      t.string :tenant_id, null: false
      t.string :rating, null: false
      t.text :comment, null: false
      t.timestamps
    end
    create_table :golden_candidates do |t|
      t.references :answer_message, null: false, foreign_key: true, index: {unique: true}
      t.string :tenant_id, null: false
      t.string :conversation_id, null: false
      t.string :prompt_version, null: false
      t.json :retrieved_chunk_ids, null: false
      t.json :payload, null: false
      t.string :status, null: false, default: 'queued'
      t.string :draft_key
      t.timestamps
    end
    add_index :golden_candidates, [:tenant_id, :status]
  end
end
