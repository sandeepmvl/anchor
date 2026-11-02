class CreateGoldenDatasets < ActiveRecord::Migration[8.1]
  def up
    create_table :golden_datasets do |t|
      t.string :tenant_id, null: false
      t.string :kind, null: false
      t.string :key, null: false
      t.json :payload, null: false
      t.timestamps
    end
    add_index :golden_datasets, [:tenant_id, :kind, :key], unique: true
    add_check_constraint :golden_datasets, "kind IN ('example','version')", name: 'golden_kind'
    install_guards!(connection)
  end

  def install_guards!(database)
    database.execute <<~SQL
      CREATE TRIGGER IF NOT EXISTS golden_version_no_update BEFORE UPDATE ON golden_datasets
      WHEN OLD.kind = 'version'
      BEGIN SELECT RAISE(ABORT, 'immutable golden version'); END;
    SQL
    database.execute <<~SQL
      CREATE TRIGGER IF NOT EXISTS golden_version_no_delete BEFORE DELETE ON golden_datasets
      WHEN OLD.kind = 'version'
      BEGIN SELECT RAISE(ABORT, 'immutable golden version'); END;
    SQL
    database.execute <<~SQL
      CREATE TRIGGER IF NOT EXISTS golden_kind_no_change BEFORE UPDATE ON golden_datasets
      WHEN OLD.kind != NEW.kind OR OLD.tenant_id != NEW.tenant_id OR OLD.key != NEW.key
      BEGIN SELECT RAISE(ABORT, 'immutable golden identity'); END;
    SQL
  end

  def down
    drop_table :golden_datasets
  end
end
