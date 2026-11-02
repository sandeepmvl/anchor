class CreateAiOperatingModes < ActiveRecord::Migration[8.1]
  def change
    create_table :ai_operating_modes do |t|
      t.references :organization, null: false, foreign_key: true, index: { unique: true }
      t.boolean :human_only, null: false, default: false
      t.bigint :revision, null: false, default: 0
      t.json :audit, null: false, default: []
      t.timestamps
    end
    add_check_constraint :ai_operating_modes, 'revision >= 0', name: 'ai_mode_revision_nonnegative'
  end
end
