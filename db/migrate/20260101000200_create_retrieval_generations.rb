class CreateRetrievalGenerations < ActiveRecord::Migration[8.1]
  def change
    create_table :retrieval_generations do |t|
      t.integer :tenant_id, null: false
      t.string :model, null: false
      t.string :revision, null: false
      t.integer :dimensions, null: false
      t.string :status, null: false, default: 'building'
      t.boolean :active, null: false, default: false
      t.string :vector_column, null: false, default: 'embedding_v2'
      t.integer :base_generation_id
      t.integer :epoch, null: false, default: 0
      t.timestamps null: false, default: -> { 'CURRENT_TIMESTAMP' }
    end
    add_index :retrieval_generations, :tenant_id, unique: true,
      where: 'active = 1', name: 'one_active_retrieval_generation'
    add_check_constraint :retrieval_generations, 'dimensions > 0', name: 'positive_retrieval_dimensions'
    add_check_constraint :retrieval_generations,
      "vector_column IN ('embedding', 'embedding_v2')", name: 'known_vector_column'
    add_column :documents, :metadata_hash, :string, null: false, default: ''
    add_column :documents, :metadata, :text, null: false, default: '{}'
    add_column :documents, :chunker_version, :string, null: false, default: 'sentence-v1'
    add_column :documents, :source_scope, :string
    add_reference :document_chunks, :retrieval_generation,
      foreign_key: { on_delete: :cascade }
    add_column :document_chunks, :embedding_v2, :text
    add_column :document_chunks, :source_hash, :string
    add_column :document_chunks, :chunker_version, :string
    add_column :document_chunks, :metadata, :text, null: false, default: '{}'
    indexes(:document_chunks).each do |index|
      if index.unique && index.columns == %w[document_id chunk_index]
        remove_index :document_chunks, name: index.name
      end
    end
    add_index :document_chunks, [:document_id, :retrieval_generation_id, :chunk_index],
      unique: true, name: 'unique_generation_chunk'
    add_index :document_chunks, [:document_id, :chunk_index], unique: true,
      where: 'retrieval_generation_id IS NULL', name: 'unique_legacy_chunk'
    # Database cascades must work for Python SQL as well as Rails callbacks.
    foreign_keys(:document_chunks).each do |key|
      if key.to_table == 'documents'
        remove_foreign_key :document_chunks, name: key.name
      end
    end
    add_foreign_key :document_chunks, :documents, on_delete: :cascade
  end
end
