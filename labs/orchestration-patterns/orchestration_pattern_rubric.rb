require 'json'
require 'yaml'

root = File.expand_path('../..', __dir__)
config = YAML.safe_load(File.read(File.join(root, 'config/orchestration.yml')))
ticket = JSON.parse(File.read(ARGV.fetch(0)))
ratings = ticket.fetch('ratings')
fields = %w[interdependence verification ordered independent handoff]
raise 'Provide exactly the five rubric fields' unless ratings.keys.sort == fields.sort
raise 'Ratings must be integers from 0 to 2' unless ratings.values.all? { |v| v.is_a?(Integer) && (0..2).cover?(v) }

rubric = config.fetch('rubric')
scores = {}
%w[supervisor-worker sequential parallel network/handoff].each do |pattern|
  scores[pattern] = rubric.fetch(pattern).sum do |field, weight|
    ratings.fetch(field) * weight
  end
end
coordination = scores.fetch('supervisor-worker')
scores.delete('supervisor-worker') if coordination < rubric.fetch('supervisor_threshold')
maximum = scores.values.max
winners = scores.select { |_pattern, score| score == maximum }.keys
if maximum.zero?
  selected = 'single-agent'
else
  raise 'Tied positive scores: clarify requirements' unless winners.length == 1
  selected = winners.first
end
puts selected
puts "Raw coordination score: #{coordination}"
raise 'Unexpected design label' unless selected == ticket.fetch('expected_pattern')
route = config.fetch('routes').fetch(ticket.fetch('ticket_type'))
raise 'Design ratings disagree with runtime route' unless selected == route
puts 'Ratings agree with runtime route'
