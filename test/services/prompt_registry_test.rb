require 'test_helper'
require 'json'

class PromptRegistryTest < ActiveSupport::TestCase
  test 'exact resolution and repeatable publication' do
    before = PromptRegistry.resolve('chat:v1')
    2.times { load Rails.root.join('db/seeds.rb') }
    { 'chat:v1' => PromptRegistry::V1, 'chat:v2' => PromptRegistry::V2 }.each do |version, tuple|
      assert_equal tuple.stringify_keys.merge('version' => version), PromptRegistry.resolve(version)
    end
    assert_equal before, PromptRegistry.resolve('chat:v1')
    assert_equal 2, PromptVersion.where(version: %w[chat:v1 chat:v2]).count
    assert_raises(ActiveRecord::RecordNotFound) { PromptRegistry.resolve('chat:missing') }
  end

  test 'published rows cannot be edited' do
    PromptRegistry.resolve('chat:v1')
    row = PromptVersion.find_by!(version: 'chat:v1')
    assert_raises(ActiveRecord::ReadOnlyRecord) { row.update!(max_tokens: 999) }
    assert_raises(ArgumentError) do
      PromptRegistry.publish('chat:v1', PromptRegistry::V2)
    end
    assert_equal PromptRegistry::V1.stringify_keys.merge('version' => 'chat:v1'), PromptRegistry.resolve('chat:v1')
  end

  test 'unknown selection does not publish a substitute' do
    before = PromptVersion.count
    assert_raises(ActiveRecord::RecordNotFound) { PromptRegistry.resolve('chat:v999') }
    assert_equal before, PromptVersion.count
  end

  test 'export contracts for offline boundary demo' do
    path = ENV['ANCHOR_PROMPT_EXPORT']
    next unless path
    contracts = %w[chat:v1 chat:v2].map { |version| PromptRegistry.resolve(version) }
    File.write(path, JSON.generate(contracts))
  end
end
