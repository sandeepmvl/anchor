require 'active_record'
require 'active_support/core_ext/hash/keys'
require 'minitest/autorun'

# Rails owns its connection pool and transaction lifecycle during the full suite.
if defined?(Rails.application) && Rails.application
  require_relative '../test_helper'
  registry_export_test_base = ActiveSupport::TestCase
else
  # Standalone smoke execution owns this connection for the whole process.
  unless defined?(ApplicationRecord)
    class ApplicationRecord < ActiveRecord::Base
      self.abstract_class = true
    end
  end
  ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: ':memory:')
  ActiveRecord::Schema.define do
    create_table :prompt_versions do |t|
      t.string :version
      t.text :template
      t.string :model_id
      t.float :temperature
      t.integer :max_tokens
    end
    add_index :prompt_versions, :version, unique: true
  end
  registry_export_test_base = Minitest::Test
end

require_relative '../../app/models/prompt_version'
require_relative '../../app/services/prompt_registry'

class PromptRegistryExportTest < registry_export_test_base
  def setup
    super
    # Rails rolls this back after each test. Standalone runs reset only this pair.
    PromptVersion.where(version: %w[chat:v1 chat:v2]).delete_all
  end

  def teardown
    # Never disconnect or re-establish a pool inside Rails' test transaction.
    # The standalone process releases its connection when it exits.
    super
  end

  def test_public_tags_resolve_to_stable_ids_and_hashes
    first = PromptRegistry.export_regression(tags: %w[chat:v1 chat:v2])
    assert_equal first, PromptRegistry.export_regression(tags: %w[chat:v1 chat:v2])
    first.each do |export|
      row = PromptVersion.find(export.fetch('version_id'))
      assert_equal row.version, export.fetch('tag')
      assert_equal row.regression_sha256, export.fetch('sha256')
      assert_equal row.regression_contract, export.fetch('contract')
    end
    puts JSON.generate(first.map { |p| p.slice('tag', 'version_id', 'sha256') })
  end

  def test_unknown_or_private_tags_are_denied
    count_before = PromptVersion.count
    assert_raises(SecurityError) do
      PromptRegistry.export_regression(tags: ['tenant-private:v1', 'chat:v2'])
    end
    # Other lessons may have seeded unrelated versions; denial must create none.
    assert_equal count_before, PromptVersion.count
  end

  def test_private_content_under_a_public_name_is_not_exported
    PromptVersion.create!(PromptRegistry::V1.merge(version: 'chat:v1',
      template: 'Private tenant instructions. Ticket: {{ticket}}'))
    assert_raises(SecurityError) do
      PromptRegistry.export_regression(tags: %w[chat:v1 chat:v2])
    end
  end
end
