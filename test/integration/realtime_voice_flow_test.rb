require 'minitest/autorun'
require 'tmpdir'
require 'open3'
require_relative '../../app/services/realtime_session_issuer'

class RealtimeVoiceBoundaryContractTest < Minitest::Test
  def test_ruby_credential_is_redeemed_once_by_python
    Dir.mktmpdir do |directory|
      path = File.join(directory, 'ledger.sqlite3')
      issuer = RealtimeSessionIssuer.new(path: path, origin: 'http://testserver',
        authorize: ->(_session, _ticket) { { 'tenant_id' => 1, 'user_id' => 3, 'ticket_id' => 11 } })
      credential = issuer.issue(session: {}, ticket_id: 11, origin: 'http://testserver')[:credential]
      program = <<~PYTHON
        import json,sys
        from anchor_ai.realtime import Ledger
        body=json.load(sys.stdin)
        ledger=Ledger(body['path'])
        scope=ledger.redeem(body['token'],'http://testserver','ruby-flow')
        assert scope == {'tenant_id':1,'user_id':3,'ticket_id':11}
        try:
            ledger.redeem(body['token'],'http://testserver','replay')
        except PermissionError:
            print('PASS Ruby -> Python ticket scope; replay rejected')
        else:
            raise AssertionError('credential replay accepted')
      PYTHON
      output, error, status = Open3.capture3('python3', '-c', program,
        stdin_data: JSON.generate(path: path, token: credential))
      assert status.success?, error
      assert_includes output, 'replay rejected'
    end
  end
end
