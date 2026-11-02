# Works directly with Ruby or through bin/rails runner.
root = File.expand_path("../..", __dir__)
python = ENV.fetch("ANCHOR_PYTHON", "python3")
puts "Anchor provider resilience: deterministic offline drill"
puts "Not verified: real provider cancellation/billing, multi-host circuit authority, ingress queue capacity."
args = [python, "-m", "pytest", "-q", "-s",
        "anchor_ai/tests/test_provider_resilience.py",
        "anchor_ai/tests/test_provider_routing.py"]
$stdout.flush
ok = system(*args, chdir: root)
abort "Provider resilience drill failed" unless ok
