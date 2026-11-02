require 'open3'
require 'timeout'

class DocumentIngestionClient
  class Error < StandardError; end

  def stage(tenant_id:, text:, suffix: '.txt')
    invoke(['stage', '--tenant', Integer(tenant_id).to_s, '--suffix', suffix], text).fetch('reference')
  end

  def ingest(tenant_id:, reference:)
    invoke(['ingest', '--tenant', Integer(tenant_id).to_s, '--reference', reference])
  end

  private

  def invoke(arguments, input = '')
    database = ENV.fetch('ANCHOR_DB_PATH') { ActiveRecord::Base.connection_db_config.database.to_s }
    environment = { 'ANCHOR_DB_PATH' => File.expand_path(database, Rails.root), 'PYTHONPATH' => Rails.root.to_s }
    output = nil
    Open3.popen3(environment, ENV.fetch('ANCHOR_PYTHON', 'python3'), '-m', 'anchor_ai.ingestion',
                 *arguments, chdir: Rails.root.to_s) do |stdin, stdout, stderr, waiter|
      errors = Thread.new { stderr.read }
      begin
        Timeout.timeout(6) do
          stdin.write(input)
          stdin.close
          output = stdout.read
          raise Error, 'ingestion_failed' unless waiter.value.success?
        end
      rescue Timeout::Error
        Process.kill('KILL', waiter.pid) rescue nil
        waiter.join
        raise Error, 'ingestion_timeout'
      ensure
        stdin.close unless stdin.closed?
        errors.join
      end
    end
    JSON.parse(output)
  rescue Error
    raise
  rescue StandardError
    raise Error, 'ingestion_failed'
  end
end
