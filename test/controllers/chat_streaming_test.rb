if ENV['ANCHOR_STREAM_SMOKE'] == '1'
  require 'minitest/autorun'
  require 'active_support/all'
  require 'logger'
  require 'ostruct'

  # Only framework plumbing is replaced. relay and telemetry are production code.
  module ActionController
    module Live
      class ClientDisconnected < IOError; end
    end
  end

  class ApplicationController
    class << self
      def protect_from_forgery(**); end
      def skip_forgery_protection(**); end
      def around_action(*); end
    end
    attr_accessor :response
  end

  module Rails
    def self.logger
      @logger ||= Logger.new(File::NULL)
    end
  end

  class LlmClient
    class BoundaryError < StandardError
      attr_reader :code
    end
    class << self
      attr_accessor :source
    end
    def stream(message, &writer)
      self.class.source.call(message, writer)
    end
  end

  require_relative '../../app/services/ai_telemetry'
  require_relative '../../app/controllers/chat_controller'

  class ChatStreamSmokeTest < Minitest::Test
    def setup
      @records = []
      AiTelemetry.sink = ->(record) { @records << record }
      @controller = ChatController.new
      @packets = []
      packets = @packets
      writer = Object.new
      writer.define_singleton_method(:write) { |packet| packets << packet }
      @controller.response = OpenStruct.new(stream: writer)
    end

    def test_heartbeat_is_first_byte_not_first_token
      LlmClient.source = ->(_message, writer) { writer.call(": ping\n\n") }
      AiTelemetry.span('http.chat') { @controller.send(:relay, 'refund') }
      record = @records.last
      assert_operator record.fetch(:first_byte_write_ms), :>=, 0
      refute record.key?(:first_token_write_ms)
      assert_equal ": ping\n\n", @packets.first
    end

    def test_status_is_not_text_and_token_has_separate_boundary
      AiTelemetry.span('http.chat') do |item|
        @controller.send(:write_stream_packet, "event: status\ndata: Preparing your reply\n\n")
        refute item.record.key?(:first_token_write_ms)
        @controller.send(:write_stream_packet, "event: token\ndata: Refunds\n\n", token: true)
      end
      record = @records.last
      assert_operator record.fetch(:first_token_write_ms), :>=, record.fetch(:first_byte_write_ms)
    end

    def test_disconnect_unwinds_upstream_generation
      closed = false
      continued = false
      LlmClient.source = lambda do |_message, writer|
        begin
          writer.call("event: token\ndata: Refunds\n\n")
          continued = true
        ensure
          closed = true
        end
      end
      @controller.response.stream.define_singleton_method(:write) { |_packet| raise IOError, 'disconnected' }
      AiTelemetry.span('http.chat') { @controller.send(:relay, 'refund') }
      assert closed, 'upstream ensure did not run'
      refute continued, 'generation continued after failed client write'
      refute @records.last.key?(:first_token_write_ms)
    end
  end
else
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
      assert_equal ['status', 'token', 'token', 'final'], response.body.scan(/^event: (.+)$/).flatten
      assert_includes response.body, text
      post '/chat', params: {message: 'Hello'}, as: :json
      assert_response :success
      assert_equal({'reply' => LlmClient::CANNED_REPLY}, response.parsed_body)
    end

    test 'early EOF and read timeout emit error not final' do
      [nil, LlmClient::BoundaryError.new('ai_read_timeout', 504)].each do |failure|
        source([token('Refunds ')], failure)
        stream_post
        assert_equal ['status', 'token', 'error'], response.body.scan(/^event: (.+)$/).flatten
      end
    end

    test 'runtime error and truncation remain distinct terminals' do
      source([token('Refunds '), "event: error\ndata: {}\n\n"])
      stream_post
      assert_equal ['status', 'token', 'error'], response.body.scan(/^event: (.+)$/).flatten
      source([token('Refunds '), final('Refunds ', true)])
      stream_post
      assert_equal ['status', 'token', 'final'], response.body.scan(/^event: (.+)$/).flatten
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
end
