require_relative '../../../labs/golden-dataset-construction-and-maintenance/test_helper'

class GoldenVersionerTest < GoldenTestCase
  def publish(name, against, evidence = docs)
    @versioner.publish!(name: name, against: against, docs: evidence, built_against_tag: 'test-release')
  end

  def approve_all
    @builder.approve!(keys: GoldenDataset.examples.where.not(key: 'seed-shipping').pluck(:key), reviewer: 'reviewer-a')
  end

  def test_v2_has_five_reviewed_additions_and_a_stable_hash
    baseline
    @builder.call(replay)
    pending = publish('pending-check', 'v1')
    assert_equal 1, pending.payload['rows'].length
    approve_all
    v2 = publish('v2', 'v1')
    expected = (1..5).map { |n| "release-run-#{n}" }
    assert_equal({'added' => expected, 'changed' => [], 'removed' => []}, v2.payload['diff'])
    assert v2.payload['rows'].all? { |row| row['review_status'] == 'approved' }
    hash = Digest::SHA256.hexdigest(GoldenDataset.canonical(v2.payload['rows']))
    assert_equal hash, v2.payload['sha256']
    assert_equal v2.id, publish('v2', 'v1').id
    assert_equal hash, publish('same-content', 'v1').payload['sha256']
    assert_equal 0, @builder.call(replay)
    assert_equal hash, v2.reload.payload['sha256']
    puts "Same support question: pending -> reviewer-a approved -> v2; diff has #{expected.length} additions; SHA256 #{hash}"
  end

  def test_edits_and_deletions_require_review_then_report_changes_and_removal
    baseline
    @builder.call(replay)
    approve_all
    v2 = publish('v2', 'v1')
    frozen = v2.payload.deep_dup
    edited = docs.deep_dup
    edited['policy']['version'] = 2
    edited['policy']['text'] = 'Refund window: 14 days.'
    edited.delete('condition')
    flagged = @versioner.revalidate!(docs: edited)
    assert_equal (1..5).map { |n| "release-run-#{n}" }, flagged
    assert_raises(ArgumentError) { publish('blocked', 'v2', edited) }
    refute GoldenDataset.versions.exists?(key: 'blocked')
    @builder.deprecate!(key: 'release-run-2', reviewer: 'reviewer-b')
    %w[release-run-1 release-run-3 release-run-4 release-run-5].each do |key|
      old = GoldenDataset.examples.find_by!(key: key).payload['expected_output']
      @builder.relabel!(key: key, label: old.gsub('30 days', '14 days'), docs: edited, reviewer: 'reviewer-b')
    end
    v3 = publish('v3', 'v2', edited)
    assert_equal [], v3.payload['diff']['added']
    assert_equal %w[release-run-1 release-run-3 release-run-4 release-run-5], v3.payload['diff']['changed']
    assert_equal ['release-run-2'], v3.payload['diff']['removed']
    assert_equal frozen, v2.reload.payload
    assert_includes v3.payload['rows'].find { |r| r['id'] == 'release-run-4' }['expected_output'], 'abstain'
    puts 'Edited and deleted evidence blocked publication; relabel/deprecate preserved history and produced the exact v3 diff.'
  end

  def test_missing_or_wrong_tenant_current_evidence_fails_closed
    baseline
    [nil, {}, {'policy' => {'tenant_id' => 'other', 'version' => 1, 'text' => 'private'}}].each do |evidence|
      assert_raises(ArgumentError) { publish('not-released', 'v1', evidence) }
    end
    refute GoldenDataset.versions.exists?(key: 'not-released')
  end

  def test_coverage_reports_overlapping_tags_and_difficulty
    @builder.call(replay)
    report = @versioner.coverage.fetch('pending')
    assert_equal 5, report['count']
    assert_equal({'easy' => 3, 'medium' => 1, 'hard' => 1}, report['difficulty'])
    assert_equal 3, report['tags']['refund']
    assert_equal 1, report['tags']['screenshot']
  end
end
