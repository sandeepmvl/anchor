require 'minitest/autorun'
require 'tmpdir'
require_relative '../../../app/services/realtime_session_issuer'

class RealtimeSessionsIssuerContractTest < Minitest::Test
  def setup
    @directory = Dir.mktmpdir
    @path = File.join(@directory, 'realtime.sqlite3')
    @authorize = lambda do |session, ticket|
      unless session[:member] == true && session[:participant] == true &&
          session[:user_id] == 3 && ticket.to_s == '11'
        raise RealtimeSessionIssuer::Denied
      end
      { 'tenant_id' => 1, 'user_id' => 3, 'ticket_id' => 11 }
    end
    @issuer = RealtimeSessionIssuer.new(path: @path,
      origin: 'http://testserver', authorize: @authorize)
  end

  def teardown
    FileUtils.remove_entry(@directory)
  end

  def issue(**changes)
    @issuer.issue(session: { member: true, participant: true, user_id: 3 }.merge(changes),
      ticket_id: 11, origin: 'http://testserver')
  end

  def test_authorized_participant_receives_hashed_short_lived_ticket_scope
    result = issue(tenant_id: 999)
    assert_operator result[:credential].length, :>=, 32
    assert_operator result[:expires_at], :<=, Time.now.to_i + 30
    db = SQLite3::Database.new(@path)
    row = db.get_first_row('SELECT digest,scope,origin,used FROM realtime_credentials')
    assert_equal Digest::SHA256.hexdigest(result[:credential]), row[0]
    assert_equal({ 'tenant_id' => 1, 'user_id' => 3, 'ticket_id' => 11 }, JSON.parse(row[1]))
    assert_equal 'http://testserver', row[2]
    assert_equal 0, row[3]
    db.close
  end

  def test_membership_and_participation_are_both_required
    assert_raises(RealtimeSessionIssuer::Denied) { issue(member: false) }
    assert_raises(RealtimeSessionIssuer::Denied) { issue(participant: false) }
    assert_raises(RealtimeSessionIssuer::Denied) { issue(user_id: 4) }
  end

  def test_other_origin_and_missing_authorizer_fail_closed
    assert_raises(RealtimeSessionIssuer::Denied) do
      @issuer.issue(session: {}, ticket_id: 11, origin: 'https://evil.invalid')
    end
    denied = RealtimeSessionIssuer.new(path: @path, origin: 'http://testserver', authorize: nil)
    assert_raises(RealtimeSessionIssuer::Denied) do
      denied.issue(session: {}, ticket_id: 11, origin: 'http://testserver')
    end
  end
end
