require 'test_helper'
require 'securerandom'

class Api::DocumentChunksControllerTest < ActionDispatch::IntegrationTest
  setup do
    @previous_tokens = ENV['ANCHOR_CHUNK_TOKENS']
    ENV['ANCHOR_CHUNK_TOKENS'] = JSON.generate('query-engine-test-secret' => 1)
    @headers = { 'Authorization' => 'Bearer query-engine-test-secret' }
    @own_document = insert_record(Document, tenant_id: 1, title: 'EU policy')
    @foreign_document = insert_record(Document, tenant_id: 2, title: 'Foreign policy')
    @own = insert_record(DocumentChunk, document_id: @own_document,
      tenant_id: 1, text: 'EU refunds require an order number.')
    @foreign = insert_record(DocumentChunk, document_id: @foreign_document,
      tenant_id: 2, text: 'Foreign tenant confidential policy.')
  end

  teardown do
    ENV['ANCHOR_CHUNK_TOKENS'] = @previous_tokens
  end

  test 'hydrates only authorized database rows' do
    get '/api/document_chunks', params: { ids: [@own.to_s] }, headers: @headers
    assert_response :success
    payload = response.parsed_body
    assert_equal 1, payload.fetch('tenant_id')
    row = payload.fetch('chunks').sole
    assert_equal @own, row.fetch('id')
    assert_equal @own_document, row.fetch('document_id')
    assert_equal 'EU refunds require an order number.', row.fetch('text')
    assert row.fetch('retrieval_metadata').is_a?(Hash)
  end

  test 'rejects another tenant ID without returning partial results' do
    get '/api/document_chunks', params: { ids: [@own.to_s, @foreign.to_s] }, headers: @headers
    assert_response :not_found
    assert_not_includes response.body, 'order number'
    assert_not_includes response.body, 'confidential'
  end

  test 'a tenant parameter cannot change authenticated scope' do
    get '/api/document_chunks', params: { ids: [@foreign.to_s], tenant_id: 2 }, headers: @headers
    assert_response :forbidden
  end

  test 'rejects missing credentials' do
    get '/api/document_chunks', params: { ids: [@own.to_s] }
    assert_response :unauthorized
  end

  test 'rejects malformed and duplicate identifiers' do
    get '/api/document_chunks', params: { ids: ['1 OR 1=1'] }, headers: @headers
    assert_response :bad_request
    get '/api/document_chunks', params: { ids: [@own.to_s, @own.to_s] }, headers: @headers
    assert_response :bad_request
  end

  private

  def insert_record(model, values)
    attributes = values.stringify_keys.slice(*model.column_names)
    model.columns.each do |column|
      next if column.name == model.primary_key || attributes.key?(column.name)
      next if column.null || !column.default.nil?
      attributes[column.name] = case column.type
      when :integer, :bigint then 1
      when :float, :decimal then 0
      when :boolean then false
      when :datetime, :timestamp then Time.current
      when :date then Date.current
      when :json, :jsonb then {}
      else "query-engine-test-#{SecureRandom.hex(8)}"
      end
    end
    model.insert_all!([attributes])
    model.order(model.primary_key => :desc).pick(model.primary_key)
  end
end
