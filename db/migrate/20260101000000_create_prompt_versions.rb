class CreatePromptVersions < ActiveRecord::Migration[8.1]
  def up
    create_table :prompt_versions do |t|
      t.string :version, null: false
      t.text :template, null: false
      t.string :model_id, null: false
      t.float :temperature, null: false
      t.integer :max_tokens, null: false
      t.timestamps
    end
    add_index :prompt_versions, :version, unique: true
    execute <<~SQL
      CREATE TRIGGER prompt_versions_no_update BEFORE UPDATE ON prompt_versions
      BEGIN SELECT RAISE(ABORT, 'published prompt versions are immutable'); END;
    SQL
    execute <<~SQL
      CREATE TRIGGER prompt_versions_no_delete BEFORE DELETE ON prompt_versions
      BEGIN SELECT RAISE(ABORT, 'published prompt versions are immutable'); END;
    SQL
  end

  def down
    drop_table :prompt_versions
  end
end
