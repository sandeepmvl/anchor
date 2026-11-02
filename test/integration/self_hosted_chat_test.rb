require 'test_helper'
require_relative 'ai_quota_enforcement_test'
require_relative 'security_tenant_isolation_test'

# Reuse the application's authenticated HTTP fixtures and assertions, not a
# second controller implementation. Inherited test methods run in these suites.
module SelfHostedSelection
  def before_setup
    @self_hosted_environment = %w[LLM_BACKEND ANCHOR_VLLM_TRANSPORT].to_h do |key|
      [key, ENV[key]]
    end
    ENV['LLM_BACKEND'] = 'self_hosted'
    ENV['ANCHOR_VLLM_TRANSPORT'] = 'offline'
    super
  end

  def after_teardown
    super
  ensure
    @self_hosted_environment&.each do |key, value|
      value.nil? ? ENV.delete(key) : ENV[key] = value
    end
  end
end

class SelfHostedChatQuotaTest < AiQuotaEnforcementTest
  include SelfHostedSelection

  def test_self_hosted_remains_behind_the_rails_ai_service_boundary
    client = LlmClient.new(provider: 'stub')
    assert_equal 'http', client.instance_variable_get(:@provider)
    assert_equal 'self_hosted', client.instance_variable_get(:@backend)
  end
end

class SelfHostedChatIsolationTest < SecurityTenantIsolationTest
  include SelfHostedSelection
end
