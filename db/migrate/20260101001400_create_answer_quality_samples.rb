class CreateAnswerQualitySamples < ActiveRecord::Migration[8.1]
  def change
    create_table :answer_quality_samples do |t|
      t.bigint :organization_id, null: false
      t.string :message_id, null: false
      t.string :answer_version, null: false
      t.string :evaluator_version, null: false
      t.string :intent, null: false
      t.string :model_version, null: false
      t.string :trace_id, null: false
      t.datetime :event_at, null: false
      t.date :budget_day, null: false
      t.json :redacted_payload, null: false
      t.string :status, null: false, default: 'pending'
      t.float :score
      t.bigint :feedback_event_id
      t.datetime :feedback_at
      t.timestamps
    end
    add_index :answer_quality_samples,
      [:organization_id, :message_id, :answer_version, :evaluator_version],
      unique: true, name: 'unique_answer_quality_version'
    add_index :answer_quality_samples, [:organization_id, :budget_day]
  end
end
