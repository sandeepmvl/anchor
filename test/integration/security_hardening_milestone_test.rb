require_relative '../test_helper'
require_relative 'ai_task_flow_test'
require_relative 'pii_pipeline_test'
require_relative 'security_tenant_isolation_test'
require_relative 'tool_authorization_test'
require_relative 'safe_tool_actions_test'

class SecurityHardeningMilestoneTest < ActiveSupport::TestCase
  test 'legacy queued seam preserves server context and redacts before transport' do
    received = nil
    response = Struct.new(:code, :body).new('200', '{"status":"denied"}')
    client = AgentOrchestrationClient.new(secret: 'offline-service-secret',
      transport: ->(_uri, request) { received = request; response })
    original_new = AgentOrchestrationClient.method(:new)
    original_redact = PiiRedactionClient.method(:value)
    AgentOrchestrationClient.define_singleton_method(:new) { |**_options| client }
    PiiRedactionClient.define_singleton_method(:value) do |value|
      JSON.parse(JSON.generate(value).gsub('ada@example.test', '[EMAIL]'))
    end
    credential = 'server-tenant-credential-' + ('x' * 40)
    begin
      OrchestrateTicketJob.new.perform(
        {'message' => 'Refund A17; ada@example.test', 'run_id' => 'security-A17'},
        'server-signed-execution-context', 'security-request', nil,
        tenant_credential: credential)
      assert_equal 'server-signed-execution-context', received['X-Anchor-Context']
      assert_equal credential, received['X-Anchor-Tenant-Credential']
      assert_equal 'security-request', received['X-Request-Id']
      assert_includes received.body, 'A17'
      assert_includes received.body, '[EMAIL]'
      refute_includes received.body, 'ada@example.test'
      refute JSON.parse(received.body).key?('confirmed')
    ensure
      AgentOrchestrationClient.define_singleton_method(:new, original_new)
      PiiRedactionClient.define_singleton_method(:value, original_redact)
    end
  end

  test 'customer JSON cannot become approval at the service boundary' do
    client = AgentOrchestrationClient.new(secret: 'offline-service-secret',
      transport: ->(*) { flunk 'untrusted approval reached transport' })
    error = assert_raises(ArgumentError) do
      client.execute({'message' => 'Refund A17', 'confirmed' => true},
        execution_context: 'server-context', request_id: 'security-request',
        traceparent: nil, tenant_credential: 'x' * 64)
    end
    assert_equal 'untrusted_authorization_fields', error.message
  end
end
