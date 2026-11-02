require 'yaml'
require 'json'
require 'optparse'

LABELS = {
  'state' => 'state-model expressiveness',
  'debug' => 'debuggability/observability',
  'ecosystem' => 'ecosystem maturity',
  'exit-cost' => 'exit-cost',
  'runtime-fit' => 'runtime-fit'
}.freeze

def validate!(data)
  raise 'Unsupported policy version' unless data.fetch('version') == 1
  candidates = %w[LangChain LangGraph LlamaIndex AutoGen provider-native]
  raise 'Wrong candidates' unless data.fetch('frameworks').keys.sort == candidates.sort
  data.fetch('frameworks').each_value do |scores|
    raise 'Wrong score axes' unless scores.keys.sort == LABELS.keys.sort
    raise 'Invalid scores' unless scores.values.all? do |v|
      v.is_a?(Numeric) && v.finite? && v.between?(1, 5)
    end
  end
  raise 'Wrong use cases' unless data.fetch('usecases').keys.sort == %w[escalation rag triage]
  data.fetch('usecases').each_value do |profile|
    {'weights' => %w[state debug ecosystem lock-in],
     'subweights' => %w[exit-cost runtime-fit]}.each do |key, axes|
      values = profile.fetch(key)
      raise 'Wrong weight axes' unless values.keys.sort == axes.sort
      raise 'Invalid weights' unless values.values.all? do |v|
        v.is_a?(Numeric) && v.finite? && v.between?(0, 1)
      end
      raise 'Weights must sum to one' unless (values.values.sum - 1).abs < 1e-9
    end
  end
end

def effective(profile)
  weights = profile.fetch('weights')
  weights.reject { |key, _| key == 'lock-in' }.merge(
    profile.fetch('subweights').transform_values { |v| v * weights.fetch('lock-in') })
end

def ranking(data, profile)
  priorities = effective(profile)
  data.fetch('frameworks').map do |name, scores|
    parts = priorities.to_h { |key, weight| [key, weight * scores.fetch(key)] }
    {'framework' => name, 'total' => parts.values.sum, 'breakdown' => parts}
  end.sort_by { |row| [-row.fetch('total'), row.fetch('framework')] }
end

options = {}
OptionParser.new do |parser|
  parser.on('--usecase NAME') { |v| options[:usecase] = v }
  parser.on('--json') { options[:json] = true }
end.parse!
begin
  raise 'Unexpected arguments' unless ARGV.empty?
  data = YAML.safe_load_file(File.expand_path('../../config/framework_selection.yml', __dir__))
  validate!(data)
  profile = data.fetch('usecases').fetch(options.fetch(:usecase))
  rows = ranking(data, profile)
  raise 'Selected framework differs from matrix' unless rows.first.fetch('framework') == profile.fetch('selected')
  raise 'Tied winner' unless rows[0].fetch('total') - rows[1].fetch('total') > 1e-9
  priorities = effective(profile).sort_by { |key, value| [-value, key] }.first(2)
  required = [profile.fetch('selected')] + priorities.map { |key, _| LABELS.fetch(key) }
  text = profile.fetch('justification')
  raise 'Justification missing required terms' unless required.all? { |term| text.downcase.include?(term.downcase) }
  if options[:json]
    puts JSON.generate(rows)
  else
    puts rows.first.fetch('framework')
    puts "Subject: #{profile.fetch('subject')}"
    rows.each do |row|
      parts = row.fetch('breakdown').map { |key, value| "#{key}=#{format('%.3f', value)}" }
      puts "#{row.fetch('framework')}: #{format('%.3f', row.fetch('total'))} (#{parts.join(', ')})"
    end
    priorities.each { |key, value| puts "#{LABELS.fetch(key)}: #{format('%.3f', value)}" }
    puts "Justification: #{text}"
    puts 'PASS: declared selection matches unique matrix winner'
  end
rescue StandardError => error
  warn "Matrix error: #{error.message}"
  exit 1
end
