require 'test_helper'

class DocumentTest < ActiveSupport::TestCase
  setup do
    # schema.rb omits SQLite triggers. Restore the migration's rules when
    # Rails has prepared the test database from that schema dump.
    connection = ActiveRecord::Base.connection
    %w[INSERT UPDATE].each do |event|
      connection.execute <<~SQL
        CREATE TRIGGER IF NOT EXISTS chunks_ownership_#{event.downcase}
        BEFORE #{event} ON document_chunks BEGIN
          SELECT RAISE(ABORT, 'chunk_document_tenant') WHERE NOT EXISTS (
            SELECT 1 FROM documents d WHERE d.id=NEW.document_id AND d.tenant_id=NEW.tenant_id
          );
          SELECT RAISE(ABORT, 'chunk_parent_lineage') WHERE NEW.is_leaf=1 AND NOT EXISTS (
            SELECT 1 FROM document_chunks p WHERE p.id=NEW.parent_id AND p.is_leaf=0
            AND p.document_id=NEW.document_id AND p.tenant_id=NEW.tenant_id
          );
          SELECT RAISE(ABORT, 'parent_has_children') WHERE EXISTS (
            SELECT 1 FROM document_chunks c WHERE c.parent_id=NEW.id
            AND (NEW.is_leaf<>0 OR c.document_id<>NEW.document_id OR c.tenant_id<>NEW.tenant_id)
          );
        END;
      SQL
    end
    connection.execute <<~SQL
      CREATE TRIGGER IF NOT EXISTS documents_tenant_update
      BEFORE UPDATE OF tenant_id ON documents BEGIN
        SELECT RAISE(ABORT, 'document_has_chunks') WHERE EXISTS (
          SELECT 1 FROM document_chunks c WHERE c.document_id=NEW.id AND c.tenant_id<>NEW.tenant_id
        );
      END;
    SQL
  end

  def document(tenant, external = 'upload-1')
    Document.create!(tenant_id: tenant, external_id: external, source_name: 'policy.md',
                     content_hash: "hash-#{tenant}-#{external}", text: 'Review refunds.', status: 'ready')
  end

  test 'documents and chunks require tenants and external identity is tenant scoped' do
    one = document(101)
    two = document(202)
    assert two.persisted?
    duplicate = one.dup
    duplicate.content_hash = 'different'
    assert_not duplicate.valid?
    assert duplicate.errors[:external_id].any?
    one.tenant_id = nil
    assert_not one.valid?
    chunk = DocumentChunk.new(document: two, chunk_index: 0, is_leaf: false, text: 'Review.')
    assert_not chunk.valid?
    assert chunk.errors[:tenant_id].any?
  end

  test 'models reject raw configured PII' do
    row = document(101)
    row.text = 'Contact pat@example.com or 415-555-0199.'
    assert_not row.valid?
    chunk = DocumentChunk.new(document: row, tenant_id: 101, chunk_index: 0,
                              is_leaf: false, text: 'Contact pat@example.com.')
    assert_not chunk.valid?
    assert_includes chunk.errors[:base], 'unredacted_payload'
  end

  test 'parent links cannot cross tenants or documents including direct SQL writes' do
    one = document(101)
    two = document(202)
    parent = DocumentChunk.create!(document: one, tenant_id: 101, chunk_index: 0, is_leaf: false, text: 'Review.')
    other = DocumentChunk.create!(document: two, tenant_id: 202, chunk_index: 0, is_leaf: false, text: 'Review.')
    leaf = DocumentChunk.create!(document: one, tenant_id: 101, chunk_index: 1, parent: parent,
      is_leaf: true, text: 'Review.', window_text: 'Review.', embedding: '[1.0]',
      embedding_model: 'fake', embedding_revision: 'v1', embedding_dimensions: 1)
    leaf.parent = other
    assert_not leaf.valid?
    leaf.reload
    assert_raises(ActiveRecord::StatementInvalid) do
      leaf.update_columns(parent_id: other.id)
    end
    assert_equal parent.id, leaf.reload.parent_id
    assert_raises(ActiveRecord::StatementInvalid) do
      leaf.update_columns(tenant_id: 202)
    end
    assert_equal 101, leaf.reload.tenant_id
    assert_nil parent.embedding
    same_tenant_other_document = document(101, 'upload-2')
    third = DocumentChunk.create!(document: same_tenant_other_document, tenant_id: 101,
                                 chunk_index: 0, is_leaf: false, text: 'Review.')
    assert_raises(ActiveRecord::StatementInvalid) { leaf.update_columns(parent_id: third.id) }
    assert_equal parent.id, leaf.reload.parent_id
  end
end
