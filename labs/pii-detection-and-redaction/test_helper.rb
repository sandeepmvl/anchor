ENV['RAILS_ENV'] = 'test'
require 'json'
require 'logger'
require 'tmpdir'
require 'rails'
require 'action_controller/railtie'
require 'active_record'
require 'action_dispatch/testing/integration'
require 'minitest/autorun'
require 'stringio'
require 'yaml'
require 'erb'

ROOT = File.expand_path('../..', __dir__) unless defined?(ROOT)
ENV['ANCHOR_PII_HMAC_KEY'] = 'ab' * 32
ENV['ANCHOR_LLM_PROVIDER'] = 'offline'

# Do not initialize a second Rails application inside bin/rails test.
# The isolated process uses a temporary root, so unrelated product initializers
# cannot run before this harness has supplied its explicit dependencies.
if defined?(Rails.application) && Rails.application&.initialized?
  require 'open3'
  class PiiIsolatedProcessTest < ActiveSupport::TestCase
    test 'PII HTTP and persistence integration in isolated process' do
      output, status = Open3.capture2e('ruby', File.join(ROOT, 'test/integration/pii_pipeline_test.rb'))
      assert status.success?, output
    end
  end
else
  class PiiSmokeApplication < Rails::Application
    config.root = Dir.mktmpdir('anchor-pii-http-')
    config.eager_load = false
    config.secret_key_base = 'offline-pii-smoke-' * 8
    config.hosts.clear
    config.logger = ActiveSupport::Logger.new(StringIO.new)
    config.action_dispatch.show_exceptions = :none
  end
  PiiSmokeApplication.initialize!
  Rails.application.routes.draw { post '/chat', to: 'chat#create' }
  # Product configuration remains rooted in Anchor, not the harness directory.
  Rails.application.config.root = Pathname.new(ROOT)

  ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: ':memory:')
  ActiveRecord::Schema.verbose = false
  ActiveRecord::Schema.define do
    create_table(:organizations) { |t| t.string :name }
    create_table(:tickets) { |t| t.integer :organization_id }
    create_table(:conversations) do |t|
      t.integer :organization_id
      t.integer :ticket_id
      t.json :pii_payload, default: {}, null: false
    end
  end

  class ApplicationRecord < ActiveRecord::Base
    self.abstract_class = true
  end
  class Organization < ApplicationRecord; end
  class Ticket < ApplicationRecord; end
  class ApplicationController < ActionController::Base; end
  module AiQuotaEnforcement; end

  class TicketAccessPolicy
    class Unauthenticated < StandardError; end
    def initialize(session); end
    def ticket(id) = Ticket.find(id)
  end

  class PromptRegistry
    class Message < String
      def initialize(value, contract)
        super(value)
      end
    end
    def self.resolve(selector) = {'version' => 'chat:v1'}
  end

  class LlmClient
    class BoundaryError < StandardError
      attr_reader :code
    end
    class ContextMessage < PromptRegistry::Message
      def initialize(value, contract, options)
        super(value, contract)
      end
    end
    class << self
      attr_accessor :prompts, :reply
    end
    self.prompts = []
    def complete(message)
      self.class.prompts << message.to_s
      self.class.reply
    end
  end

  require File.join(ROOT, 'app/services/pii_redaction_client')
  require File.join(ROOT, 'app/models/conversation')
  require File.join(ROOT, 'app/services/ai_telemetry')
  require File.join(ROOT, 'app/controllers/chat_controller')
  require File.join(ROOT, 'config/initializers/filter_parameter_logging')

  class ChatController
    private
    def retrieval_settings = nil
  end

  class PiiIntegrationCase < ActionDispatch::IntegrationTest
    def setup
      Conversation.delete_all
      Ticket.delete_all
      Organization.delete_all
      @organization = Organization.create!(name: 'Offline fixture')
      @ticket = Ticket.create!(organization_id: @organization.id)
      @fixture = JSON.parse(File.read(File.join(ROOT, 'anchor_ai/tests/fixtures/pii_test_fixtures.json')))
      LlmClient.prompts = []
      LlmClient.reply = @fixture.fetch('response')
      @records = []
      AiTelemetry.sink = ->(record) { @records << record }
      @log = StringIO.new
      Rails.logger = ActiveSupport::Logger.new(@log)
      PiiRedactionClient.transport = nil
    end

    def teardown
      PiiRedactionClient.transport = nil
    end
  end
end
