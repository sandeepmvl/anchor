require 'optparse'
require_relative '../app/services/finetuning/dataset_curator'

begin
  options = {}
  OptionParser.new do |parser|
    %w[tenant since as-of source out].each do |name|
      parser.on("--#{name} VALUE") { |value| options[name.tr('-', '_').to_sym] = value }
    end
  end.parse!
  required = %i[tenant since as_of source out]
  raise ArgumentError unless required.all? { |key| options.key?(key) } && ARGV.empty?
  result = Finetuning::DatasetCurator.new(**options).call
  puts JSON.pretty_generate(result.slice('tenant_id', 'counts', 'label_histograms', 'status'))
  puts 'Anchor: tenant-scoped redacted export; duplicate tickets removed; auditable grouped splits written.'
rescue StandardError
  warn 'curation_aborted: check scope, source eligibility, and redaction boundary'
  exit 1
end
