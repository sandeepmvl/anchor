require 'test_helper'

class PromptVersionTest < ActiveSupport::TestCase
  setup do
    PromptRegistry.resolve('chat:v1')
    @prompt_version = PromptVersion.find_by!(version: 'chat:v1')
    assert @prompt_version.valid?, @prompt_version.errors.full_messages.join(', ')
  end

  test 'requires every prompt contract field' do
    %i[template model_id temperature max_tokens version].each do |field|
      candidate = @prompt_version.dup
      candidate.version = 'chat:v900'
      candidate.public_send("#{field}=", nil)
      assert_not candidate.valid?, "accepted missing #{field}"
      assert candidate.errors.added?(field, :blank),
        "missing presence error for #{field}: #{candidate.errors.full_messages.inspect}"
    end
  end

  test 'version must be unique' do
    duplicate = @prompt_version.dup
    assert_not duplicate.valid?, 'accepted a duplicate version'
    assert duplicate.errors.of_kind?(:version, :taken)
  end

  test 'zero temperature is valid' do
    candidate = @prompt_version.dup
    candidate.version = 'chat:v900'
    candidate.temperature = 0
    assert candidate.valid?, candidate.errors.full_messages.join(', ')
  end

  test 'fresh complete prompt is valid' do
    candidate = @prompt_version.dup
    candidate.version = 'chat:v900'
    assert candidate.valid?, candidate.errors.full_messages.join(', ')
  end
end
