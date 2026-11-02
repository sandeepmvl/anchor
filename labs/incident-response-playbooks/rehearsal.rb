require 'active_record'
require 'sqlite3'
require 'json'
require 'fileutils'
require 'time'
require_relative '../../app/services/ai_incident_control'
require_relative '../../app/services/ai_quota_policy'

# Run only in a standalone process: never replace Rails models or its connection.
if defined?(Rails) && Rails.respond_to?(:application) && Rails.application
  raise 'Run the rehearsal in a separate Ruby process, not the Rails console.'
end

# Attribute type lookup needs an adapter before the model classes are defined.
# This bootstrap connection contains no application data. connect replaces it
# with the explicitly selected, isolated rehearsal database.
ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: ':memory:')

# Process-local rehearsal storage adapter, not replacements for Rails models.
class Organization < ActiveRecord::Base; end
class Membership < ActiveRecord::Base; end
class AiOperatingMode < ActiveRecord::Base
  attribute :audit, :json, default: []
end

module IncidentRehearsal
  ROOT = File.expand_path('../..', __dir__)
  DIR = File.join(ROOT, 'var/incident_playbook')
  DB = File.join(DIR, 'rehearsal.sqlite3')
  TENANT = 3760
  OTHER = 3761
  CONTEXT = {
    'ticket_id' => 'refund-3760', 'message' => 'Am I eligible for a refund?',
    'history' => ['Customer supplied purchase date'],
    'evidence' => ['redacted-order-reference'],
    'proposed_action' => 'Human verifies refund eligibility'
  }.freeze

  def self.connect(path, create: false)
    ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: path)
    return unless create
    ActiveRecord::Schema.verbose = false
    ActiveRecord::Schema.define do
      create_table(:organizations) { |t| t.string :name }
      create_table(:memberships) do |t|
        t.integer :organization_id
        t.integer :user_id
        t.string :role
      end
      create_table(:ai_operating_modes) do |t|
        t.integer :organization_id, null: false
        t.boolean :human_only, default: false, null: false
        t.integer :revision, default: 0, null: false
        t.text :audit, default: '[]', null: false
      end
      add_index :ai_operating_modes, :organization_id, unique: true
    end
    [Organization, Membership, AiOperatingMode].each(&:reset_column_information)
    Organization.create!(id: TENANT, name: 'isolated-rehearsal')
    Organization.create!(id: OTHER, name: 'unaffected-rehearsal')
    Membership.create!(organization_id: TENANT, user_id: 1, role: 'admin')
    Membership.create!(organization_id: TENANT, user_id: 2, role: 'agent')
    Membership.create!(organization_id: OTHER, user_id: 3, role: 'admin')
  end

  def self.ledger(path = DB, clock: -> { Time.now.to_f })
    AiQuotaPolicy::IncidentWindow.new(path: path, cap: 12, seconds: 3600, clock: clock)
  end

  def self.seed_spike(path = DB, clock: -> { Time.now.to_f })
    quota = ledger(path, clock: clock)
    quota.call('queue', tenant: TENANT, ticket: 'refund-3760', context: CONTEXT)
    10.times do |index|
      result = quota.call('reserve', tenant: TENANT, ticket: 'refund-3760',
        context: CONTEXT, attempt_id: "synthetic-#{index}", bound: 1)
      raise 'unexpected admission failure' unless result.fetch('allowed')
    end
    quota.incident_signal(tenant: TENANT, baseline: 1)
  end

  def self.latest
    AiOperatingMode.find_by!(organization_id: TENANT).audit.reverse.find do |event|
      event['event'] == 'incident_declared'
    end || raise('classify first')
  end

  def self.banner(path = DB)
    ticket = ledger(path).call('snapshot', tenant: TENANT).fetch('tickets').find do |row|
      row.fetch('ticket') == 'refund-3760'
    end
    AiIncidentControl.run_agent(organization_id: TENANT,
      payload: { 'request' => ticket.fetch('context') }, key: 'refund-rehearsal') do
      raise 'FAIL: autonomous dispatch reached during containment verification'
    end
  end

  def self.run(command)
    if command == 'simulate'
      raise 'existing rehearsal: archive var/incident_playbook before a new run' if File.exist?(DB)
      FileUtils.mkdir_p(DIR)
      connect(DB, create: true)
      signal = seed_spike
      event = AiIncidentControl.declare!(organization_id: TENANT, signal: signal)
      puts JSON.pretty_generate(event)
      puts 'Synthetic budget anomaly recorded; paging delivery NOT verified.'
      return
    end
    raise 'run simulate first' unless File.exist?(DB)
    connect(DB)
    event = latest
    case command
    when 'classify'
      puts JSON.pretty_generate(event)
      puts 'SEV1: operations owns immediate containment; name responders in the incident document.'
    when 'contain'
      member = Membership.find_by!(organization_id: TENANT, role: 'admin')
      mode = AiIncidentControl.contain_incident!(member: member,
        organization_id: TENANT, incident_id: event.fetch('incident_id'))
      puts JSON.pretty_generate(mode.audit.last)
    when 'chat'
      response = banner
      verified = Time.now.utc
      elapsed = verified - Time.iso8601(event.fetch('detected_at'))
      result = { 'incident_id' => event.fetch('incident_id'), 'response' => response,
        'verified_at' => verified.iso8601(6), 'elapsed_seconds' => elapsed,
        'within_60_seconds' => verified <= Time.iso8601(event.fetch('human_banner_deadline')),
        'boundary' => 'real service handoff; HTTP rendering and cancellation NOT verified' }
      File.write(File.join(DIR, 'banner.json'), JSON.pretty_generate(result))
      puts JSON.pretty_generate(result)
      raise 'deadline missed: preserve evidence and document the delay' unless result['within_60_seconds']
    when 'recover'
      member = Membership.find_by!(organization_id: TENANT, role: 'admin')
      begin
        AiIncidentControl.recover_incident!(member: member, organization_id: TENANT,
          incident_id: event.fetch('incident_id'), checks: {}, production: false)
        raise 'FAIL: recovery accepted missing checks'
      rescue AiIncidentControl::ReleaseBlocked => error
        puts "Recovery refused: #{error.message}"
      end
      raise 'contain first' unless AiIncidentControl.human_only?(TENANT)
      template = File.read(File.join(ROOT, 'ops/runbooks/ai_incidents.md')).split('## Postmortem template').last
      audit = AiOperatingMode.find_by!(organization_id: TENANT).audit
      report = "# Rehearsal postmortem — DRAFT, review required\n#{template}\n"
      report += "\n## Recorded audit\n```json\n#{JSON.pretty_generate(audit)}\n```\n"
      report += "\nSynthetic 10x reservation-bound spike; actual loss and production repair not measured.\n"
      report += "Human-only remains active. Complete names, timeline, actions, and review fields.\n"
      File.write(File.join(DIR, 'postmortem.md'), report)
      puts 'Human-only remains active. Draft: var/incident_playbook/postmortem.md'
      puts 'Production coordination, cancellation, HTTP banner, and real paging NOT verified.'
    else
      raise ArgumentError, 'expected simulate, classify, contain, chat, or recover'
    end
  end
end

IncidentRehearsal.run(ARGV.fetch(0)) if $PROGRAM_NAME == __FILE__
