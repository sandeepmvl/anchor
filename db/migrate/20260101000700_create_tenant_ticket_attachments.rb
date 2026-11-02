class CreateTenantTicketAttachments < ActiveRecord::Migration[8.1]
  def change
    create_table(:organizations) { |t| t.string :name, null: false }
    create_table :users do |t|
      t.string :email, null: false
      t.string :password_digest, null: false
    end
    add_index :users, :email, unique: true
    create_table :memberships do |t|
      t.references :organization, null: false, foreign_key: true
      t.references :user, null: false, foreign_key: true
      t.string :role, null: false
    end
    add_index :memberships, [:organization_id, :user_id], unique: true
    create_table :tickets do |t|
      t.references :organization, null: false, foreign_key: true
      t.references :user, null: false, foreign_key: true
      t.string :subject, null: false
    end
    create_table :conversations do |t|
      t.references :organization, null: false, foreign_key: true
      t.references :ticket, null: false, foreign_key: true, index: { unique: true }
      t.json :attachment_answer, default: {}, null: false
    end
    create_table :ticket_attachments do |t|
      t.references :organization, null: false, foreign_key: true
      t.references :ticket, null: false, foreign_key: true
      t.string :kind, null: false
      t.string :status, null: false
      t.string :notice
      t.json :evidence, default: [], null: false
      t.json :answer, default: {}, null: false
      t.boolean :offline_fixture, default: false, null: false
      t.timestamps
    end
    create_table :active_storage_blobs do |t|
      t.string :key, null: false
      t.string :filename, null: false
      t.string :content_type
      t.text :metadata
      t.string :service_name, null: false
      t.bigint :byte_size, null: false
      t.string :checksum
      t.datetime :created_at, null: false
      t.index :key, unique: true
    end
    create_table :active_storage_attachments do |t|
      t.string :name, null: false
      t.references :record, null: false, polymorphic: true, index: false
      t.references :blob, null: false, foreign_key: { to_table: :active_storage_blobs }
      t.datetime :created_at, null: false
      t.index [:record_type, :record_id, :name, :blob_id], unique: true,
        name: :index_attachment_uniqueness
    end
    create_table :active_storage_variant_records do |t|
      t.belongs_to :blob, null: false, foreign_key: { to_table: :active_storage_blobs }
      t.string :variation_digest, null: false
      t.index [:blob_id, :variation_digest], unique: true,
        name: :index_variant_uniqueness
    end
  end
end
