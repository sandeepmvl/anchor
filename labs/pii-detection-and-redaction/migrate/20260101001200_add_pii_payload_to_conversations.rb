class AddPiiPayloadToConversations < ActiveRecord::Migration[8.1]
  def change
    add_column :conversations, :pii_payload, :json, default: {}, null: false
  end
end
