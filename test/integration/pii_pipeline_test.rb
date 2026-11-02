require_relative '../../labs/pii-detection-and-redaction/test_helper'

if defined?(PiiIntegrationCase)
  # This standalone harness supplies its own schema rather than running Rails migrations.
  require_relative '../../app/models/ai_operating_mode'
  require_relative '../../app/services/ai_incident_control'
  unless ActiveRecord::Base.connection.table_exists?(:ai_operating_modes)
    require_relative '../../db/migrate/20260101001500_create_ai_operating_modes'
    CreateAiOperatingModes.new.migrate(:up)
  end

  class PiiPipelineTest < PiiIntegrationCase
    def setup
      # Delete child rows before the harness resets its organizations. These tests
      # exercise normal chat; incident denial is covered by human_only_mode_test.
      AiOperatingMode.delete_all
      super
    end
    def assert_no_originals(value)
      encoded = value.is_a?(String) ? value : JSON.generate(value)
      @fixture.fetch('identifiers').each { |identifier| refute_includes encoded, identifier }
      refute_includes encoded, @fixture.fetch('notes')
    end

    def test_chat_persists_and_returns_only_redacted_text
      post '/chat', params: {message: @fixture.fetch('message'),
        notes: @fixture.fetch('notes'), ticket_id: @ticket.id}, as: :json
      assert_equal 200, response.status
      assert_equal 1, Conversation.count
      payload = Conversation.first.pii_payload
      assert_includes payload.fetch('ingestion'), '[REDACTED_NAME]'
      assert_includes payload.fetch('ingestion'), '[REDACTED_EMAIL]'
      assert_includes payload.fetch('ingestion'), '[REDACTED_CARD_'
      assert_includes JSON.generate(payload.fetch('response')), '[REDACTED_CARD_'
      [payload, response.body, LlmClient.prompts, @records, @log.string].each do |value|
        assert_no_originals(value)
      end
      assert_equal 1, LlmClient.prompts.length
      assert_equal @organization.id, Conversation.first.organization_id
      puts JSON.generate(event: 'pii_demo', conversation: payload,
        prompt_trace: LlmClient.prompts.first, customer_reply: JSON.parse(response.body))
    end

    def test_service_failure_prevents_persistence_and_forwarding
      PiiRedactionClient.transport = ->(_value) { raise PiiRedactionClient::Unavailable }
      post '/chat', params: {message: @fixture.fetch('message'), ticket_id: @ticket.id}, as: :json
      assert_equal 503, response.status
      assert_equal 0, Conversation.count
      assert_empty LlmClient.prompts
      assert_no_originals(response.body)
      assert_no_originals(@records)
      assert_no_originals(@log.string)
    end

    def test_model_callback_guards_direct_persistence
      row = Conversation.create!(organization_id: @organization.id, ticket_id: @ticket.id,
        pii_payload: {'ingestion' => @fixture.fetch('message')})
      assert_no_originals(row.reload.pii_payload)
      PiiRedactionClient.transport = ->(_value) { raise PiiRedactionClient::Unavailable }
      assert_raises(PiiRedactionClient::Unavailable) do
        Conversation.create!(organization_id: @organization.id, ticket_id: @ticket.id,
          pii_payload: {'ingestion' => @fixture.fetch('message')})
      end
      assert_equal 1, Conversation.count
    end

    def test_pre_controller_parameter_filter_hides_entire_fields
      filter = ActiveSupport::ParameterFilter.new(Rails.application.config.filter_parameters)
      values = {'message' => @fixture.fetch('message'), 'history' => [@fixture.fetch('message')],
        'notes' => @fixture.fetch('notes'), 'summary_state' => {'summary' => @fixture.fetch('message')}}
      filtered = filter.filter(values)
      assert filtered.values.all? { |value| value == '[FILTERED]' }
      assert_no_originals(filtered)
    end
  end
end
