require 'active_record'
require 'sqlite3'
require 'tmpdir'
require 'fileutils'
require 'minitest/autorun'
require_relative '../../app/services/prompt_registry'

# Rails owns its model, connection, schema, and transaction lifecycle.
# Only the standalone command supplies minimal persistence scaffolding.
if defined?(Rails.application)
  require_relative '../test_helper'
else
  unless defined?(PromptVersion)
    class PromptVersion < ActiveRecord::Base
    end
  end
end

class SecurityPromptRegistryTest < (defined?(Rails.application) ? ActiveSupport::TestCase : Minitest::Test)
  def setup
    super
    @previous = ENV['ANCHOR_REPLY_PROMPT_VERSION']
    @standalone = !defined?(Rails.application)
    if @standalone
      @directory = Dir.mktmpdir('anchor-security-registry-')
      ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: File.join(@directory, 'test.sqlite3'))
      ActiveRecord::Schema.verbose = false
      ActiveRecord::Schema.define do
        create_table :prompt_versions do |t|
          t.string :version, null: false
          t.text :template, null: false
          t.string :model_id, null: false
          t.float :temperature, null: false
          t.integer :max_tokens, null: false
        end
        add_index :prompt_versions, :version, unique: true
      end
      PromptVersion.reset_column_information
    end
    ENV['ANCHOR_REPLY_PROMPT_VERSION'] = 'chat:v2'
  end

  def teardown
    ENV['ANCHOR_REPLY_PROMPT_VERSION'] = @previous
    if @standalone
      ActiveRecord::Base.connection_pool.disconnect!
      FileUtils.remove_entry(@directory) if @directory
    end
    super
  end

  def test_derivation_activation_and_rollback_preserve_versions
    old = %w[chat:v1 chat:v2 chat:v3].to_h { |v| [v, PromptRegistry.resolve(v)] }
    candidate = PromptRegistry.harden_active(version: 'chat:security-v1')
    published = PromptRegistry.resolve(candidate.version)
    assert_equal old['chat:v2']['template'] + PromptRegistry::HARDENING, published['template']
    %w[model_id temperature max_tokens].each do |field|
      assert_equal old['chat:v2'][field], published[field]
    end
    assert_equal 'chat:v2', PromptRegistry.active_reply['version']
    ENV['ANCHOR_REPLY_PROMPT_VERSION'] = candidate.version
    assert_equal published, PromptRegistry.active_reply
    ENV['ANCHOR_REPLY_PROMPT_VERSION'] = 'chat:v2'
    assert_equal old['chat:v2'], PromptRegistry.active_reply
    old.each { |version, contract| assert_equal contract, PromptRegistry.resolve(version) }
    puts 'Published chat:security-v1 from active chat:v2; prior versions unchanged; rollback passed.'
  end

  def test_existing_version_cannot_be_rewritten
    before = PromptRegistry.active_reply
    assert_raises(ArgumentError) { PromptRegistry.harden_active(version: before['version']) }
    attributes = before.except('version').merge('template' => 'replacement')
    assert_raises(ArgumentError) { PromptRegistry.publish(before['version'], attributes) }
    assert_equal before, PromptRegistry.active_reply
  end
end
