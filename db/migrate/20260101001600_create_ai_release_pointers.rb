class CreateAiReleasePointers < ActiveRecord::Migration[8.1]
  def change
    create_table :ai_release_pointers do |t|
      t.bigint :tenant_id, null: false
      t.string :channel, null: false, default: 'reply'
      t.json :bundles, null: false, default: {}
      t.string :stable, null: false
      t.string :previous
      t.string :candidate
      t.integer :canary_percent, null: false, default: 0
      t.integer :generation, null: false, default: 0
      t.string :required_contract, null: false
      t.json :audit, null: false, default: []
      t.timestamps
    end
    add_index :ai_release_pointers, [:tenant_id, :channel], unique: true
    add_check_constraint :ai_release_pointers,
      'canary_percent >= 0 AND canary_percent <= 100', name: 'ai_canary_percent'
    add_check_constraint :ai_release_pointers,
      'tenant_id > 0 AND generation >= 0', name: 'ai_release_identity'
  end
end
