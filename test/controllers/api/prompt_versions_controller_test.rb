require 'test_helper'

class Api::PromptVersionsControllerTest < ActionDispatch::IntegrationTest
  setup do
    @previous_secret = ENV['ANCHOR_AI_SECRET']
    @previous_version = ENV['ANCHOR_REPLY_PROMPT_VERSION']
    ENV['ANCHOR_AI_SECRET'] = 'registry-test-secret'
    ENV['ANCHOR_REPLY_PROMPT_VERSION'] = 'chat:v3'
  end

  teardown do
    ENV['ANCHOR_AI_SECRET'] = @previous_secret
    ENV['ANCHOR_REPLY_PROMPT_VERSION'] = @previous_version
  end

  test 'authenticated internal lookup returns the active registry contract' do
    expected = PromptRegistry.active_reply
    get '/api/prompt_versions/active', headers: {
      'X-Anchor-Secret' => 'registry-test-secret'
    }
    assert_response :success
    assert_equal expected, response.parsed_body
    assert_equal 'chat:v3', response.parsed_body.fetch('version')
    assert_includes response.parsed_body.fetch('template'), '{{ticket}}'
  end

  test 'unauthenticated lookup is rejected without a prompt' do
    get '/api/prompt_versions/active'
    assert_response :unauthorized
    assert_equal({ 'detail' => 'service_auth_failed' }, response.parsed_body)
  end

  test 'incorrect credentials are rejected' do
    get '/api/prompt_versions/active', headers: { 'X-Anchor-Secret' => 'wrong' }
    assert_response :unauthorized
    assert_not response.parsed_body.key?('template')
  end

  test 'missing server configuration fails closed' do
    ENV.delete('ANCHOR_AI_SECRET')
    get '/api/prompt_versions/active'
    assert_response :service_unavailable
    assert_equal({ 'detail' => 'service_not_configured' }, response.parsed_body)
  end
end
