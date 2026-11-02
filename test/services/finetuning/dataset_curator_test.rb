require 'minitest/autorun'
require 'tmpdir'
require 'rbconfig'
require_relative '../../../app/services/finetuning/dataset_curator'

class DatasetCuratorTest < Minitest::Test
  SOURCE = File.join(Finetuning::DatasetCurator::ROOT, 'eval/finetune/fixtures/transcripts.jsonl')

  def setup
    @dir = Dir.mktmpdir
    @out = File.join(@dir, 'export', 'train.jsonl')
    @args = {tenant: 'demo', since: '30d', as_of: '2026-01-31', source: SOURCE, out: @out}
  end

  def teardown
    FileUtils.remove_entry(@dir)
  end

  def test_explicit_tenant_and_authorization_are_required
    assert_raises(ArgumentError) { Finetuning::DatasetCurator.new(**@args.merge(tenant: nil)) }
    assert_raises(ArgumentError) { Finetuning::DatasetCurator.new(**@args.merge(tenant: 'other')).call }
    refute File.exist?(@out)
  end

  def test_rejects_ineligible_rows_and_removes_duplicates
    report = Finetuning::DatasetCurator.new(**@args).call
    expected = {'wrong_tenant' => 1, 'outside_window' => 1, 'review_deferred' => 1,
                'pii_deferred' => 1, 'length' => 1, 'exact_duplicate' => 1,
                'normalized_duplicate' => 1, 'near_duplicate' => 1, 'accepted' => 6}
    assert_equal expected, report['counts']
    ids = report['audit'].map { |entry| entry['id'] }
    refute_includes ids, 'other'
    rows = report['files'].values.flat_map { |path| File.readlines(path).map { |line| JSON.parse(line) } }
    assert_equal 6, rows.length
    assert rows.all? { |row| row.keys.sort == %w[input label output] }
    text = JSON.generate(rows)
    refute_includes text, 'customer@example.test'
    refute_includes text, 'support@example.test'
    refute_includes text, '212-555-0199'
    assert_includes text, '[REDACTED_EMAIL]'
    assert rows.any? { |row| row['input'].include?('12 days') }
    assert rows.any? { |row| row['input'].include?('45 days') }
  end

  def test_related_conversations_and_duplicate_families_share_a_split
    report = Finetuning::DatasetCurator.new(**@args).call
    entries = report['audit'].select { |entry| entry.key?('split') }
    %w[conversation_id family_id group_id].each do |key|
      entries.group_by { |entry| entry[key] }.each_value do |group|
        assert_equal 1, group.map { |entry| entry['split'] }.uniq.length
      end
    end
    by_id = entries.to_h { |entry| [entry['id'], entry] }
    assert_equal by_id['b']['split'], by_id['e']['split']
    assert_equal 1, %w[a a2 a3 a4].map { |id| by_id[id]['group_id'] }.uniq.length
    assert_equal %w[held_out train validation], entries.map { |e| e['split'] }.uniq.sort
    before = File.read(@out.sub('.jsonl', '.manifest.json'))
    Finetuning::DatasetCurator.new(**@args).call
    assert_equal before, File.read(@out.sub('.jsonl', '.manifest.json'))
  end

  def test_redaction_process_failure_cannot_produce_an_export
    previous_rubyopt = ENV['RUBYOPT']
    # Load through the installed interpreter: temporary files need not be executable.
    # Consume stdin before exiting so Open3 can finish writing the entire batch.
    failure_script = File.join(@dir, 'fail_redaction.rb')
    File.write(failure_script, "STDIN.read\nexit 1\n")
    ENV['RUBYOPT'] = "-r#{failure_script}"
    error = assert_raises(RuntimeError) do
      Finetuning::DatasetCurator.new(**@args.merge(python: RbConfig.ruby)).call
    end
    assert_equal 'finetuning_redaction_failed', error.message
    refute Dir.exist?(File.dirname(@out))
  ensure
    if previous_rubyopt.nil?
      ENV.delete('RUBYOPT')
    else
      ENV['RUBYOPT'] = previous_rubyopt
    end
  end

  def test_conflicting_targets_abort_instead_of_teaching_both
    rows = File.readlines(SOURCE).map { |line| JSON.parse(line) }
    rows.find { |row| row['id'] == 'a2' }['output'] = 'A contradictory reviewed response that needs another human decision.'
    source = File.join(@dir, 'transcripts.jsonl')
    File.write(source, rows.map { |row| JSON.generate(row) + "\n" }.join)
    FileUtils.cp(SOURCE.sub('.jsonl', '_manifest.json'), source.sub('.jsonl', '_manifest.json'))
    assert_raises(RuntimeError) { Finetuning::DatasetCurator.new(**@args.merge(source: source)).call }
    refute File.exist?(@out)
  end
end
