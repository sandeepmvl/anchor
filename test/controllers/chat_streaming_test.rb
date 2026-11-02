require 'test_helper'

class ChatStreamingTest < ActionDispatch::IntegrationTest
  setup do
    @complete = LlmClient.instance_method(:complete)
    @stream = LlmClient.instance_method(:stream)
    LlmClient.define_method(:complete) { |_message| LlmClient::CANNED_REPLY }
  end

  teardown do
    LlmClient.define_method(:complete, @complete)
    LlmClient.define_method(:stream, @stream)
  end

  def source(packets, failure = nil)
    LlmClient.define_method(:stream) do |_message, &writer|
      packets.each { |packet| writer.call(packet) }
      raise failure if failure
    end
  end

  def token(text)
    "event: token\ndata: #{text}\n\n"
  end

  def final(text, truncated = false)
    "event: final\ndata: #{JSON.generate(content: text, citations: [], truncated: truncated)}\n\n"
  end

  def stream_post
    post '/chat', params: {message: 'When will my refund arrive?'}, as: :json,
      headers: {'Accept' => 'text/event-stream'}
  end

  test 'SSE preserves token order and one final while JSON keeps its contract' do
    text = 'Refunds take five days. ✓'
    bytes = (token('Refunds ') + token('take five days. ✓') + final(text)).b
    source(bytes.bytes.map { |b| b.chr })
    stream_post
    assert_response :success
    assert_equal 'text/event-stream', response.media_type
    assert_equal ['token', 'token', 'final'], response.body.scan(/^event: (.+)$/).flatten
    assert_includes response.body, text
    post '/chat', params: {message: 'Hello'}, as: :json
    assert_response :success
    assert_equal({'reply' => LlmClient::CANNED_REPLY}, response.parsed_body)
  end

  test 'early EOF and read timeout emit error not final' do
    [nil, LlmClient::BoundaryError.new('ai_read_timeout', 504)].each do |failure|
      source([token('Refunds ')], failure)
      stream_post
      assert_equal ['token', 'error'], response.body.scan(/^event: (.+)$/).flatten
    end
  end

  test 'runtime error and truncation remain distinct terminals' do
    source([token('Refunds '), "event: error\ndata: {}\n\n"])
    stream_post
    assert_equal ['token', 'error'], response.body.scan(/^event: (.+)$/).flatten
    source([token('Refunds '), final('Refunds ', true)])
    stream_post
    assert_equal ['token', 'final'], response.body.scan(/^event: (.+)$/).flatten
    assert_includes response.body, '"truncated":true'
  end

  test 'streaming POST requires CSRF when forgery protection is enabled' do
    previous = ActionController::Base.allow_forgery_protection
    ActionController::Base.allow_forgery_protection = true
    stream_post
    assert_response :unprocessable_entity
  ensure
    ActionController::Base.allow_forgery_protection = previous
  end
end
