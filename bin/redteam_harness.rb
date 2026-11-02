# Baseline permission is checked here and again in the Python worker.
if ARGV.include?('--baseline') &&
    !(ENV['RAILS_ENV'] == 'test' && ENV['ANCHOR_REDTEAM_ISOLATED'] == '1')
  abort 'baseline_requires_isolated_test_process'
end
Dir.chdir(File.expand_path('..', __dir__))
exec('python3', '-m', 'eval.redteam_gate', *ARGV)
