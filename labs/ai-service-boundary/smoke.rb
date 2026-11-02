require "action_dispatch/testing/integration"
require "json"

$stdout.sync = true
Rails.configuration.x.llm[:provider] = "http"
session = ActionDispatch::Integration::Session.new(Rails.application)
message = { message: "Can I return my order?" }
session.post("/chat", params: message, as: :json)
expected = { "reply" => "Offline support reply: please check the tenant's return policy." }
unless session.response.status == 200 && JSON.parse(session.response.body) == expected
  raise "Unexpected live runtime response: #{session.response.status} #{session.response.body}"
end
puts "PASS: same chat message crossed Rails -> Python; #{session.response.body}"
puts "STOP_RUNTIME"
raise "Missing shutdown confirmation" unless $stdin.gets&.strip == "STOPPED"
session.post("/chat", params: message, as: :json)
unless session.response.status == 503 &&
    JSON.parse(session.response.body) == { "error" => "ai_connection_refused" }
  raise "Unexpected unavailable response: #{session.response.status} #{session.response.body}"
end
puts "PASS: unavailable runtime -> controlled 503 #{session.response.body}"
