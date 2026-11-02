#!/usr/bin/env ruby
# Run the Python AI boundary without booting Rails or duplicating its policy.
require 'rbconfig'
root = File.expand_path('..', __dir__)
Dir.chdir(root) do
  python = ENV.fetch('PYTHON', 'python3')
  success = system({ 'ANCHOR_LLM_PROVIDER' => 'offline' }, python,
                   '-m', 'anchor_ai.mcp', *ARGV)
  exit(success ? 0 : 1)
end
