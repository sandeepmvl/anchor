require 'rails'
require 'active_record'
require 'action_controller/railtie'
require 'action_controller/test_case'
require 'minitest/autorun'
require 'pathname'

module ScopedAuthorizationLab
  ROOT = File.expand_path('../..', __dir__)
  class Application < Rails::Application
    config.eager_load = false
    config.secret_key_base = 'offline-authorization-test-secret-' * 4
    config.logger = Logger.new(File::NULL)
    config.hosts.clear
  end
end
Rails.application = ScopedAuthorizationLab::Application.instance
Rails.application.routes.draw do
  post '/api/tool_actions', to: 'api/tool_actions#create'
end
ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: ':memory:')
ActiveRecord::Schema.verbose = false
ActiveRecord::Schema.define do
  create_table(:organizations) { |t| t.string :name }
  create_table(:users) do |t|
    t.string :name
    t.string :email, null: false
    t.string :password_digest, null: false
  end
  create_table :memberships do |t|
    t.references :organization, null: false
    t.references :user, null: false
    t.string :role, null: false
  end
end
class ApplicationRecord < ActiveRecord::Base
  self.abstract_class = true
end
class ApplicationController < ActionController::Base; end
module Api; end
root = ScopedAuthorizationLab::ROOT
%w[app/models/user.rb app/models/organization.rb app/models/membership.rb app/models/order.rb
   db/migrate/20260101001200_create_tenant_orders.rb
   app/services/ticket_access_policy.rb app/services/tool_authorization_policy.rb
   app/controllers/api/tool_actions_controller.rb].each do |path|
  require File.join(root, path)
end
ActiveRecord::Migration.suppress_messages { CreateTenantOrders.new.migrate(:up) }
puts 'Offline controller harness: browser login, cookie transport, payment provider, and deployment routing NOT verified.'
