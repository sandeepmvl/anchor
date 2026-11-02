require 'json'
require 'open3'
require_relative '../../app/services/ai_quota_policy'

if ARGV.include?('--rpc')
  begin
    p = JSON.parse(STDIN.read)
    policy = JSON.parse(File.read(File.expand_path('../../config/ai_budgets.yml', __dir__))).fetch('incident_window')
    authority = AiQuotaPolicy::IncidentWindow.new(path: p.fetch('database'),
      cap: policy.fetch('cap_units'), seconds: policy.fetch('window_seconds'),
      clock: -> { p.fetch('now', 1893456000) })
    arguments = p.fetch('arguments').transform_keys(&:to_sym)
    puts JSON.generate(authority.call(p.fetch('operation'), **arguments))
  rescue StandardError => e
    puts JSON.generate('error' => e.message)
    exit 1
  end
else
  root = File.expand_path('../..', __dir__)
  mode = ARGV.find { |arg| %w[broken validate corrected human all].include?(arg) } || 'all'
  output, error, status = Open3.capture3(
    'python3', '-c',
    'from anchor_ai.runtime import incident_main; incident_main()', mode,
    chdir: root)
  puts output
  warn error unless error.empty?
  exit status.exitstatus unless status.success?
end
