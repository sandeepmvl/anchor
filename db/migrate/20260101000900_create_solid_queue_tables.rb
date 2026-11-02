class CreateSolidQueueTables < ActiveRecord::Migration[8.1]
  def up
    return if table_exists?(:solid_queue_jobs)
    load Rails.root.join("db/queue_schema.rb")
  end

  def down
    raise ActiveRecord::IrreversibleMigration, "Queue tables may contain accepted work"
  end
end
