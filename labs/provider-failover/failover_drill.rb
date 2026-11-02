# Works with ruby directly or through the existing Rails runner.
root = File.expand_path('../..', __dir__)
script = File.join(__dir__, 'drill.py')
mode = ARGV.first || 'outage'
ok = system('python3', script, mode, chdir: root)
raise 'provider failover drill failed' unless ok
