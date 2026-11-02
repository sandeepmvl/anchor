class CreateTenantOrders < ActiveRecord::Migration[8.1]
  def change
    create_table :orders do |t|
      t.references :organization, null: false, foreign_key: true
      t.string :reference, null: false
      t.string :payment_id, null: false
      t.string :status, null: false, default: 'delivered'
      t.boolean :refund_eligible, null: false, default: true
      t.integer :refund_limit_cents, null: false, default: 500
      t.integer :refunded_cents, null: false, default: 0
      t.string :refund_key
      t.timestamps
    end
    add_index :orders, [:organization_id, :reference], unique: true
    add_check_constraint :orders, 'refund_limit_cents > 0', name: 'positive_refund_limit'
    add_check_constraint :orders, 'refunded_cents >= 0', name: 'nonnegative_refund'
  end
end
