require 'test_helper'
require 'open3'

class RetrievalClientTest < ActiveSupport::TestCase
  def test_server_context_wins_over_submitted_tenant_and_history
    submitted = { tenant_id: 2, conversation_id: 200, history: ['another tenant ticket'] }
    context = RetrievalClient::TrustedContext.new(tenant_id: 1, conversation_id: 10)
    observed = nil
    transport = lambda do |payload, headers|
      observed = [payload, headers]
      { 'reply' => 'Refunds require review. Contact support with your receipt.',
        'citations' => [101], 'abstained' => false }
    end
    client = RetrievalClient.new(context: context, transport: transport)
    result = client.complete('How do I get a refund?', request_id: 'rails-request-3672',
                             traceparent: '00-' + 'a' * 32 + '-' + 'b' * 16 + '-01')
    assert_equal 1, observed[0][:tenant_id]
    assert_equal 10, observed[0][:conversation_id]
    refute observed[0].key?(:history)
    refute_equal submitted[:tenant_id], observed[0][:tenant_id]
    assert_equal 'rails-request-3672', observed[1]['X-Request-Id']
    assert_equal '00-' + 'a' * 32 + '-' + 'b' * 16 + '-01', observed[1]['traceparent']
    assert_equal [101], result['citations']
  end

  def test_missing_or_untrusted_tenant_context_is_rejected_before_transport
    [nil, { tenant_id: 1, conversation_id: 10 }].each do |context|
      assert_raises(RetrievalClient::Error) do
        RetrievalClient.new(context: context, transport: ->(*) { flunk 'must not send' })
      end
    end
    assert_raises(RetrievalClient::Error) do
      RetrievalClient::TrustedContext.new(tenant_id: 0, conversation_id: 10)
    end
  end

  def test_invalid_source_reference_shape_is_rejected
    client = RetrievalClient.new(
      context: RetrievalClient::TrustedContext.new(tenant_id: 1, conversation_id: 10),
      transport: ->(*) { { 'reply' => 'Policy', 'citations' => ['raw text'], 'abstained' => false } })
    assert_raises(RetrievalClient::Error) do
      client.complete('Question', request_id: 'request-3672',
                      traceparent: '00-' + 'a' * 32 + '-' + 'b' * 16 + '-01')
    end
  end
end

class DenseChatIntegrationTest < ActionDispatch::IntegrationTest
  def test_fixture_chat_crosses_python_boundary_without_a_server
    # Keep the Live controller and global test-only HTTP bridge out of the
    # process that subsequently runs ingestion jobs and their DB assertions.
    unless ENV['ANCHOR_DENSE_CHAT_TEST_CHILD'] == '1'
      output, errors, status = Open3.capture3(
        { 'ANCHOR_DENSE_CHAT_TEST_CHILD' => '1', 'RAILS_ENV' => 'test',
          'PARALLEL_WORKERS' => '1' },
        RbConfig.ruby, Rails.root.join('bin/rails').to_s, 'test',
        'test/services/retrieval_client_test.rb', '--name',
        'test_fixture_chat_crosses_python_boundary_without_a_server',
        chdir: Rails.root.to_s)
      puts output
      warn errors unless errors.empty?
      assert status.success?, "Isolated chat integration failed:\n#{output}\n#{errors}"
      return
    end

    previous = ENV['ANCHOR_DENSE_FIXTURE']
    previous_secret = ENV['ANCHOR_AI_SECRET']
    ENV['ANCHOR_DENSE_FIXTURE'] = '1'
    ENV['ANCHOR_AI_SECRET'] = 'offline-dense-test-only'
    original = Net::HTTP.instance_method(:request)
    root = Rails.root.to_s
    Net::HTTP.define_method(:request) do |request, *args, &block|
      if request.path == '/v1/dense_rag'
        output, status = Open3.capture2('python3', File.join(root, 'bin/dense_rag_query.py'),
          '--http-payload', request.body, '--request-id', request['X-Request-Id'],
          '--traceparent', request['traceparent'], chdir: root)
        raise 'offline bridge failed' unless status.success?
        Struct.new(:code, :body).new('200', output)
      else
        original.bind_call(self, request, *args, &block)
      end
    end
    post '/chat', params: { message: 'How do I get a refund?', tenant_id: 2,
      history: [{ position: 1, role: 'user', content: 'Other tenant text' }] }, as: :json
    assert_response :success
    result = JSON.parse(response.body)
    assert_includes result.fetch('reply'), 'Refunds require review.'
    assert_equal [101], result.fetch('citations')
    puts "Anchor chat: #{result.fetch('reply')} Sources: #{result.fetch('citations').inspect}"
  ensure
    # The parent never installs the bridge or changes the service environment.
    if original
      Net::HTTP.define_method(:request, original)
      previous.nil? ? ENV.delete('ANCHOR_DENSE_FIXTURE') : ENV['ANCHOR_DENSE_FIXTURE'] = previous
      previous_secret.nil? ? ENV.delete('ANCHOR_AI_SECRET') : ENV['ANCHOR_AI_SECRET'] = previous_secret
    end
  end
end

class HybridRetrievalClientTest < ActiveSupport::TestCase
  def traceparent
    '00-' + 'a' * 32 + '-' + 'b' * 16 + '-01'
  end

  def hybrid_result(tenant = 1)
    { 'tenant_id' => tenant, 'generation_id' => 'fixture-g1', 'hits' => [
      { 'id' => 102, 'tenant_id' => tenant, 'text' => 'Synthetic order excerpt',
        'score' => 1.0 / 61 + 1.0 / 62, 'dense' => 1.0 / 62, 'sparse' => 1.0 / 61 }
    ] }
  end

  def test_hybrid_http_propagates_trusted_scope_and_trace
    calls = []
    result = hybrid_result
    original = Net::HTTP.instance_method(:request)
    Net::HTTP.define_method(:request) do |request, *args|
      calls << request
      Struct.new(:code, :body).new('200', JSON.generate(result))
    end
    client = RetrievalClient.new(
      context: RetrievalClient::TrustedContext.new(tenant_id: 1, conversation_id: 10),
      url: 'http://127.0.0.1:8000/v1/dense_rag', secret: 'test-secret')
    assert_equal result, client.retrieve('order 4821', request_id: 'hybrid-rails', traceparent: traceparent)
    assert_equal 1, calls.length
    request = calls.first
    assert_equal '/v1/retrieve/hybrid', request.path
    assert_equal 'test-secret', request['X-Anchor-Secret']
    assert_equal 'hybrid-rails', request['X-Request-Id']
    assert_equal traceparent, request['traceparent']
    assert_equal({ 'query' => 'order 4821', 'tenant_id' => 1,
                   'conversation_id' => 10, 'top_k' => 3 }, JSON.parse(request.body))
  ensure
    Net::HTTP.define_method(:request, original) if original
  end

  def test_hybrid_failure_never_retries_unscoped
    calls = []
    original = Net::HTTP.instance_method(:request)
    Net::HTTP.define_method(:request) do |request, *args|
      calls << request
      Struct.new(:code, :body).new('503', '{}')
    end
    client = RetrievalClient.new(
      context: RetrievalClient::TrustedContext.new(tenant_id: 1, conversation_id: 10),
      url: 'http://127.0.0.1:8000/v1/dense_rag', secret: 'test-secret')
    assert_raises(RetrievalClient::Error) do
      client.retrieve('order 4821', request_id: 'hybrid-rails', traceparent: traceparent)
    end
    assert_equal 1, calls.length
    assert_equal '/v1/retrieve/hybrid', calls.first.path
    assert_equal 1, JSON.parse(calls.first.body)['tenant_id']
  ensure
    Net::HTTP.define_method(:request, original) if original
  end

  def test_hybrid_rejects_foreign_tenant_without_fallback
    calls = []
    transport = lambda do |payload, headers|
      calls << [payload, headers]
      hybrid_result(2)
    end
    client = RetrievalClient.new(
      context: RetrievalClient::TrustedContext.new(tenant_id: 1, conversation_id: 10),
      transport: transport)
    assert_raises(RetrievalClient::Error) do
      client.retrieve('order 4821', request_id: 'hybrid-rails', traceparent: traceparent)
    end
    assert_equal 1, calls.length
    assert_equal 1, calls.first[0][:tenant_id]
    refute calls.first[0].key?(:history)
  end
end
