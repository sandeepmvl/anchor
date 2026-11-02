require 'sqlite3'
require 'json'
require 'digest'
require 'securerandom'

class RealtimeSessionIssuer
  class Denied < StandardError; end

  # The product adapter must reuse TicketAccessPolicy and return this scope only
  # after checking both organization membership and participation in this ticket.
  # It receives the authenticated session, never a client-supplied tenant scope.
  def initialize(path:, origin:, authorize:)
    @path, @origin, @authorize = path, origin, authorize
    raise Denied unless origin.is_a?(String) && origin.start_with?('http://', 'https://')
  end

  def issue(session:, ticket_id:, origin:)
    raise Denied unless origin == @origin && @authorize.respond_to?(:call)
    scope = @authorize.call(session, ticket_id)
    keys = %w[tenant_id user_id ticket_id]
    raise Denied unless scope.is_a?(Hash) && scope.keys.sort == keys.sort
    raise Denied unless scope.values.all? { |value| value.is_a?(Integer) && value.positive? }
    raise Denied unless scope['ticket_id'].to_s == ticket_id.to_s
    credential = SecureRandom.urlsafe_base64(32)
    expires = Time.now.to_i + 30
    db = SQLite3::Database.new(@path)
    begin
      db.busy_timeout = 200
      db.execute(<<~SQL)
        CREATE TABLE IF NOT EXISTS realtime_credentials (
          digest TEXT PRIMARY KEY, scope TEXT NOT NULL, origin TEXT NOT NULL,
          expires REAL NOT NULL, used INTEGER NOT NULL DEFAULT 0)
      SQL
      db.execute('INSERT INTO realtime_credentials VALUES(?,?,?,?,0)',
        [Digest::SHA256.hexdigest(credential), JSON.generate(scope), origin, expires])
    ensure
      db.close
    end
    { credential: credential, expires_at: expires, websocket_path: '/v1/realtime' }
  rescue SQLite3::Exception
    raise Denied
  end

  def self.from_application
    config = Rails.application.config.x
    new(path: ENV.fetch('ANCHOR_REALTIME_LEDGER'),
      origin: ENV.fetch('ANCHOR_REALTIME_ORIGIN'),
      authorize: config.realtime_authorizer)
  rescue KeyError
    raise Denied
  end
end
