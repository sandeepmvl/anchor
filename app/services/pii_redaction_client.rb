require 'json'
require 'open3'
require 'timeout'
require 'net/http'
require 'uri'
require 'securerandom'

class PiiRedactionClient
  LOCAL_KEY = SecureRandom.hex(32).freeze

  class Unavailable < StandardError
    def initialize
      super('pii_redaction_unavailable')
    end
    def code = 'pii_redaction_unavailable'
    def status = 503
  end

  class << self
    attr_accessor :transport

    def value(value)
      if transport
        raise Unavailable if defined?(Rails) && Rails.env.production?
        return transport.call(value)
      end
      new.call(value)
    end
  end

  def call(value)
    input = JSON.generate(value: value)
    raise Unavailable if input.bytesize > 262_144
    output = production? ? remote(input) : local(input)
    result = JSON.parse(output)
    raise Unavailable unless result.is_a?(Hash) && result.keys == ['value']
    result.fetch('value')
  rescue StandardError
    raise Unavailable, cause: nil
  end

  private

  def production?
    defined?(Rails) && Rails.env.production?
  end

  def remote(input)
    uri = URI(ENV.fetch('ANCHOR_PII_URL'))
    secret = ENV.fetch('ANCHOR_AI_SECRET')
    raise Unavailable unless uri.scheme == 'https' && uri.host && uri.userinfo.nil? &&
      uri.query.nil? && uri.fragment.nil? && uri.path == '/v1/pii/redact' && !secret.empty?
    request = Net::HTTP::Post.new(uri)
    request['Content-Type'] = 'application/json'
    request['X-Anchor-Secret'] = secret
    AiTelemetry.current&.headers&.each { |key, value| request[key] = value }
    request.body = input
    output = +''
    Timeout.timeout(5) do
      Net::HTTP.start(uri.host, uri.port, use_ssl: true, open_timeout: 2,
        read_timeout: 3, write_timeout: 3) do |http|
        http.request(request) do |response|
          raise Unavailable unless response.code == '200'
          response.read_body do |chunk|
            output << chunk
            raise Unavailable if output.bytesize > 1_048_576
          end
        end
      end
    end
    output
  end

  def local(input)
    root = File.expand_path('../..', __dir__)
    command = [ENV.fetch('ANCHOR_PYTHON', 'python3'),
      File.join(root, 'labs/pii-detection-and-redaction/worker.py')]
    environment = {'ANCHOR_PII_HMAC_KEY' => ENV.fetch('ANCHOR_PII_HMAC_KEY', LOCAL_KEY)}
    output = nil
    Open3.popen3(environment, *command, chdir: root) do |stdin, stdout, stderr, wait|
      readers = []
      begin
        Timeout.timeout(5) do
          readers = [stdout, stderr].map do |io|
            Thread.new do
              Thread.current.report_on_exception = false
              bytes = io.read(1_048_577) || ''
              raise Unavailable if bytes.bytesize > 1_048_576
              bytes
            end
          end
          stdin.write(input)
          stdin.close
          output = readers[0].value
          readers[1].value
          raise Unavailable unless wait.value.success?
        end
      ensure
        stdin.close unless stdin.closed?
        if wait.alive?
          begin
            Process.kill('KILL', wait.pid)
          rescue Errno::ESRCH
          end
        end
        wait.value
        readers.each(&:join)
      end
    end
    output
  end
end
