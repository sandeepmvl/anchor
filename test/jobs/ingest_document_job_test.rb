require 'test_helper'
require 'tmpdir'

class IngestDocumentJobTest < ActiveJob::TestCase
  self.use_transactional_tests = false

  setup do
    @old_stage = ENV['ANCHOR_STAGE_ROOT']
    @old_database = ENV['ANCHOR_DB_PATH']
    @old_provider = ENV['ANCHOR_INGESTION_EMBEDDINGS']
    @directory = Dir.mktmpdir('anchor-staging')
    ENV['ANCHOR_STAGE_ROOT'] = @directory
    ENV['ANCHOR_INGESTION_EMBEDDINGS'] = 'offline'
    ENV['ANCHOR_DB_PATH'] = File.expand_path(ActiveRecord::Base.connection_db_config.database, Rails.root)
    DocumentChunk.delete_all
    Document.delete_all
    clear_enqueued_jobs
    clear_performed_jobs
  end

  teardown do
    DocumentChunk.where(is_leaf: true).delete_all
    DocumentChunk.delete_all
    Document.delete_all
    FileUtils.remove_entry(@directory)
    ENV['ANCHOR_STAGE_ROOT'] = @old_stage
    ENV['ANCHOR_DB_PATH'] = @old_database
    ENV['ANCHOR_INGESTION_EMBEDDINGS'] = @old_provider
  end

  test 'queue carries only tenant and staged reference and repeated jobs are idempotent' do
    raw = 'Refunds require review. Contact pat@example.com for help.'
    reference = DocumentIngestionClient.new.stage(tenant_id: 101, text: raw)
    assert_match(/\A[0-9a-f]{32}\z/, reference)
    assert_not_includes File.read(File.join(@directory, "#{reference}.json")), 'pat@example.com'
    IngestDocumentJob.perform_later(101, reference)
    assert_equal [101, reference], enqueued_jobs.last[:args]
    assert_not_includes enqueued_jobs.inspect, raw
    perform_enqueued_jobs
    assert_equal 'ready', Document.find_by!(tenant_id: 101).status
    before = [Document.count, DocumentChunk.count]
    IngestDocumentJob.perform_now(101, reference)
    assert_equal before, [Document.count, DocumentChunk.count]
    assert_not_includes Document.first.text, 'pat@example.com'
  end

  test 'safe failure queues a reference-only retry and later succeeds' do
    reference = DocumentIngestionClient.new.stage(tenant_id: 101, text: 'Review refunds.')
    valid_database = ENV.fetch('ANCHOR_DB_PATH')
    ENV['ANCHOR_DB_PATH'] = File.join(@directory, 'missing.sqlite3')
    IngestDocumentJob.perform_now(101, reference)
    assert_equal 1, enqueued_jobs.length
    assert_equal [101, reference], enqueued_jobs.last[:args]
    assert_equal 0, Document.count
    ENV['ANCHOR_DB_PATH'] = valid_database
    retry_payload = enqueued_jobs.fetch(0).select { |key, _value| key.is_a?(String) }
    clear_enqueued_jobs
    ActiveJob::Base.execute(retry_payload)
    assert_equal 0, enqueued_jobs.length, 'the recovered retry must not enqueue another failure'
    # Python's separate connection cannot invalidate Rails' cached zero count.
    ActiveRecord::Base.uncached do
      assert_equal 1, Document.count
      assert_equal 'ready', Document.first.status
    end
  end
end
