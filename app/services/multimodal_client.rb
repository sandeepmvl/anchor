require 'base64'
require 'json'
require 'openssl'
require 'open3'
require 'securerandom'
require 'timeout'

class MultimodalClient
  class Unavailable < StandardError; end
  MAX_BYTES = 2_000_000
  AUDIO_MAX_BYTES = 24_000_000

  def self.limit_for(kind)
    kind == 'voice' ? AUDIO_MAX_BYTES : MAX_BYTES
  end

  def extract(raw, kind:, tenant_id:, conversation_id:)
    raise Unavailable unless %w[screenshot scanned_page voice].include?(kind)
    raise Unavailable unless raw.bytesize.between?(1, self.class.limit_for(kind))
    invoke({ operation: 'extract', bytes: Base64.strict_encode64(raw), kind: kind },
      tenant_id: tenant_id, conversation_id: conversation_id)
  end

  def answer(evidence, tenant_id:, conversation_id:)
    invoke({ operation: 'answer', evidence: evidence },
      tenant_id: tenant_id, conversation_id: conversation_id)
  end

  def realtime_session(session:, ticket_id:, origin:)
    RealtimeSessionIssuer.from_application.issue(
      session: session, ticket_id: ticket_id, origin: origin)
  end

  private

  def invoke(body, tenant_id:, conversation_id:)
    secret = SecureRandom.hex(32)
    claims = { aud: 'anchor-vision-v1', tenant_id: tenant_id,
      conversation_id: conversation_id, exp: Time.now.to_i + 60 }
    encoded = Base64.urlsafe_encode64(JSON.generate(claims), padding: false)
    signature = OpenSSL::HMAC.hexdigest('SHA256', secret, encoded)
    input = JSON.generate(body.merge(context: "#{encoded}.#{signature}",
      tenant_id: tenant_id, conversation_id: conversation_id))
    env = { 'ANCHOR_ATTACHMENT_PROCESS_SECRET' => secret,
      'ANCHOR_ATTACHMENT_PRODUCTION' => Rails.env.production? ? '1' : '0' }
    result = nil
    Open3.popen3(env, 'python3', '-m', 'anchor_ai.attachment_evidence',
      chdir: Rails.root.to_s, pgroup: true) do |stdin, stdout, stderr, wait|
      begin
        Timeout.timeout(5) do
          stdin.write(input)
          stdin.close
          result = stdout.read(65_537)
          raise Unavailable if result.bytesize > 65_536
          raise Unavailable unless wait.value.success?
        end
      rescue Exception
        begin
          Process.kill('KILL', -wait.pid)
        rescue Errno::ESRCH
          nil
        end
        raise
      ensure
        stdin.close unless stdin.closed?
        stdout.close
        stderr.close
      end
    end
    value = JSON.parse(result)
    raise Unavailable unless value.is_a?(Hash) && !value.key?('error')
    value
  rescue StandardError
    raise Unavailable, 'attachment_service_unavailable'
  end
end
