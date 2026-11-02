require 'test_helper'

class Api::AgentMemoriesControllerTest < ActionDispatch::IntegrationTest
  setup do
    @previous_credentials = ENV['ANCHOR_MEMORY_CREDENTIALS']
    ENV['ANCHOR_MEMORY_CREDENTIALS'] = { 'worker-a' => 3709, 'worker-b' => 3710 }.to_json
    AgentMemory.where(tenant_id: [3709, 3710]).delete_all
  end

  teardown do
    if @previous_credentials
      ENV['ANCHOR_MEMORY_CREDENTIALS'] = @previous_credentials
    else
      ENV.delete('ANCHOR_MEMORY_CREDENTIALS')
    end
  end

  def headers_for(token = 'worker-a')
    { 'Authorization' => "Bearer #{token}" }
  end

  def payload
    { conversation_id: 123, run_key: 'refund-123-v1',
      summary: 'Contact sam@example.com or +1 415 555 0123. Manual approval required.' }
  end

  test 'memory endpoints require authentication' do
    assert_no_difference('AgentMemory.count') do
      post '/api/agent_memories', params: payload, as: :json
      assert_response :unauthorized
    end
    get '/api/agent_memories', params: { conversation_id: 123 }
    assert_response :unauthorized
  end

  test 'writes are redacted and retries preserve one row' do
    assert_difference('AgentMemory.count', 1) do
      post '/api/agent_memories', params: payload, headers: headers_for, as: :json
      assert_response :success
      post '/api/agent_memories', params: payload.merge(summary: 'retry replacement'),
        headers: headers_for, as: :json
      assert_response :success
    end
    row = AgentMemory.find_by!(tenant_id: 3709, conversation_id: 123)
    assert_includes row.summary, '[EMAIL]'
    assert_includes row.summary, '[PHONE]'
    refute_includes row.summary, 'sam@example.com'
    refute_includes row.summary, '415 555 0123'
    refute_equal 'retry replacement', row.summary
  end

  test 'authenticated scope overrides forged tenant and isolates reads' do
    post '/api/agent_memories', params: payload.merge(tenant_id: 3710),
      headers: headers_for, as: :json
    assert_response :success
    assert AgentMemory.exists?(tenant_id: 3709, conversation_id: 123)
    refute AgentMemory.exists?(tenant_id: 3710, conversation_id: 123)
    get '/api/agent_memories', params: { conversation_id: 123 },
      headers: headers_for('worker-b')
    assert_response :success
    assert_equal [], response.parsed_body.fetch('memories')
    post '/api/agent_memories', params: payload,
      headers: headers_for('worker-b'), as: :json
    assert_response :success
    assert_equal 2, AgentMemory.where(conversation_id: 123,
      tenant_id: [3709, 3710]).count
  end

  test 'invalid scope creates no row' do
    assert_no_difference('AgentMemory.count') do
      post '/api/agent_memories', params: payload.merge(conversation_id: 0),
        headers: headers_for, as: :json
      assert_response :unprocessable_entity
    end
  end
end
