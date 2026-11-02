require 'test_helper'
require 'open3'

class SecurityTenantIsolationTest < ActionDispatch::IntegrationTest
  ROOT = File.expand_path('../..', __dir__)

  def test_chat_rejects_forged_ownership_and_restored_state_before_generation
    previous = ENV['ANCHOR_TENANT_CREDENTIALS']
    ENV['ANCHOR_TENANT_CREDENTIALS'] = JSON.generate('offline-tenant-a-credential-000000000000' => 1)
    %w[tenant_id conversation_id history summary_state checkpoint_id].each do |field|
      post '/chat', params: {message: 'What is our refund policy?', field => 'B_ONLY_COPPER_MOON'}, as: :json
      assert_response :forbidden
      assert_equal 'untrusted_tenant_state', response.parsed_body.fetch('error')
      refute_includes response.body, 'B_ONLY_COPPER_MOON'
    end
  ensure
    previous.nil? ? ENV.delete('ANCHOR_TENANT_CREDENTIALS') : ENV['ANCHOR_TENANT_CREDENTIALS'] = previous
  end

  def test_chunk_api_rejects_forged_tenant_after_service_authentication
    previous = ENV['ANCHOR_CHUNK_TOKENS']
    token = 'offline-chunk-tenant-a-token'
    ENV['ANCHOR_CHUNK_TOKENS'] = JSON.generate(token => 1)
    get '/api/document_chunks', params: {tenant_id: 2, ids: ['2']},
      headers: {'Authorization' => "Bearer #{token}"}
    assert_response :forbidden
    refute_includes response.body, 'B_ONLY_COPPER_MOON'
    get '/api/document_chunks', params: {ids: ['2']}
    assert_response :unauthorized
  ensure
    previous.nil? ? ENV.delete('ANCHOR_CHUNK_TOKENS') : ENV['ANCHOR_CHUNK_TOKENS'] = previous
  end

  def test_offline_product_components_do_not_restore_foreign_tenant_data
    output, error, status = Open3.capture3(
      'python3', 'labs/tenant-isolation/tenant_isolation_test.py', chdir: ROOT
    )
    assert status.success?, "#{output}\n#{error}"
    assert_includes output, 'unsafe global winner: tenant B'
    assert_includes output, 'tenant A candidates: 1'
    assert_includes output, 'tenant B chunks: 0'
    assert_includes output, 'foreign conversation: absent'
    assert_includes output, 'cache: tenant-local miss'
    assert_includes output, 'checkpoint: absent'
    assert_includes output, 'mis-scoped write: rejected'
    assert_includes output, 'missing scope: rejected'
    assert_includes output, 'Rails login'
  end
end
