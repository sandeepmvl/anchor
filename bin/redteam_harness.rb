# Baseline permission is checked here and again in the Python worker.
if ARGV.include?('--baseline') &&
    !(ENV['RAILS_ENV'] == 'test' && ENV['ANCHOR_REDTEAM_ISOLATED'] == '1')
  abort 'baseline_requires_isolated_test_process'
end
Dir.chdir(File.expand_path('..', __dir__))
if ARGV.include?('--suite') && ARGV.include?('full') &&
    ARGV.include?('--target') && ARGV.include?('hardened_release')
  exec(ENV.fetch('ANCHOR_PYTHON', 'python3'), '-m', 'labs.security_hardening.milestone', *ARGV)
end
exec('python3', '-m', 'eval.redteam_gate', *ARGV)
