class CreateDocumentsAndDocumentChunks < ActiveRecord::Migration[8.1]
  def up
    create_table :documents do |t|
      t.bigint :tenant_id, null: false
      t.string :external_id, null: false
      t.string :source_name, null: false
      t.string :content_hash, null: false
      t.text :text, null: false
      t.string :status, null: false, default: 'processing'
      t.timestamps
    end
    add_index :documents, [:tenant_id, :content_hash], unique: true
    add_index :documents, [:tenant_id, :external_id], unique: true
    add_check_constraint :documents, 'tenant_id > 0', name: 'documents_tenant'
    add_check_constraint :documents, "status IN ('processing','ready')", name: 'documents_status'
    create_table :document_chunks do |t|
      t.bigint :tenant_id, null: false
      t.references :document, null: false, foreign_key: true
      t.integer :chunk_index, null: false
      t.references :parent, foreign_key: { to_table: :document_chunks }
      t.boolean :is_leaf, null: false
      t.text :text, null: false
      t.text :window_text
      t.text :embedding
      t.string :embedding_model
      t.string :embedding_revision
      t.integer :embedding_dimensions
      t.timestamps
    end
    add_index :document_chunks, [:document_id, :chunk_index], unique: true
    add_index :document_chunks, [:tenant_id, :is_leaf]
    add_check_constraint :document_chunks, 'tenant_id > 0 AND chunk_index >= 0', name: 'chunks_identity'
    add_check_constraint :document_chunks, <<~SQL.squish, name: 'chunks_payload'
      (is_leaf=0 AND parent_id IS NULL AND embedding IS NULL AND
       embedding_model IS NULL AND embedding_revision IS NULL AND embedding_dimensions IS NULL)
      OR (is_leaf=1 AND parent_id IS NOT NULL AND embedding IS NOT NULL AND
       window_text IS NOT NULL AND embedding_model IS NOT NULL AND
       embedding_revision IS NOT NULL AND embedding_dimensions > 0)
    SQL
    %w[INSERT UPDATE].each do |event|
      execute <<~SQL
        CREATE TRIGGER chunks_ownership_#{event.downcase}
        BEFORE #{event} ON document_chunks BEGIN
          SELECT RAISE(ABORT, 'chunk_document_tenant') WHERE NOT EXISTS (
            SELECT 1 FROM documents d WHERE d.id=NEW.document_id AND d.tenant_id=NEW.tenant_id
          );
          SELECT RAISE(ABORT, 'chunk_parent_lineage') WHERE NEW.is_leaf=1 AND NOT EXISTS (
            SELECT 1 FROM document_chunks p WHERE p.id=NEW.parent_id AND p.is_leaf=0
            AND p.document_id=NEW.document_id AND p.tenant_id=NEW.tenant_id
          );
          SELECT RAISE(ABORT, 'parent_has_children') WHERE EXISTS (
            SELECT 1 FROM document_chunks c WHERE c.parent_id=OLD_ID_PLACEHOLDER
            AND (NEW.is_leaf<>0 OR c.document_id<>NEW.document_id OR c.tenant_id<>NEW.tenant_id)
          );
        END;
      SQL
        .gsub('OLD_ID_PLACEHOLDER', 'NEW.id')
    end
    execute <<~SQL
      CREATE TRIGGER documents_tenant_update BEFORE UPDATE OF tenant_id ON documents BEGIN
        SELECT RAISE(ABORT, 'document_has_chunks') WHERE EXISTS (
          SELECT 1 FROM document_chunks c WHERE c.document_id=NEW.id AND c.tenant_id<>NEW.tenant_id
        );
      END;
    SQL
  end

  def down
    execute 'DROP TRIGGER IF EXISTS documents_tenant_update'
    drop_table :document_chunks
    drop_table :documents
  end
end
