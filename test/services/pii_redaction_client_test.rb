require 'minitest/autorun'
require_relative '../../app/services/pii_redaction_client'

class PiiRedactionClientTest < Minitest::Test
  class ProductionOutage < PiiRedactionClient
    attr_reader :local_called, :remote_called

    private

    def production? = true

    def remote(input)
      @remote_called = true
      raise Errno::ECONNREFUSED
    end

    def local(input)
      @local_called = true
      raise 'must not fall back'
    end
  end

  def test_local_default_needs_no_server_or_deployment_key
    previous = ENV.delete('ANCHOR_PII_HMAC_KEY')
    original_transport = PiiRedactionClient.transport
    PiiRedactionClient.transport = nil
    raw = 'Paulo Santos, email paulo@example.test, card 4111 1111 1111 1111.'
    safe = PiiRedactionClient.value(raw)
    assert_includes safe, '[REDACTED_NAME]'
    assert_includes safe, '[REDACTED_EMAIL]'
    assert_includes safe, '[REDACTED_CARD_'
    refute_includes safe, 'Paulo Santos'
    refute_includes safe, '4111 1111 1111 1111'
    assert_equal safe, PiiRedactionClient.value(raw)
  ensure
    previous.nil? ? ENV.delete('ANCHOR_PII_HMAC_KEY') : ENV['ANCHOR_PII_HMAC_KEY'] = previous
    PiiRedactionClient.transport = original_transport
  end

  def test_production_external_outage_never_uses_local_fallback
    client = ProductionOutage.new
    error = assert_raises(PiiRedactionClient::Unavailable) do
      client.call('private source text')
    end
    assert_equal 'pii_redaction_unavailable', error.message
    assert_equal 503, error.status
    assert client.remote_called
    refute client.local_called
    assert_nil error.cause
  end
end
