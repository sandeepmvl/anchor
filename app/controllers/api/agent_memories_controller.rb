class Api::AgentMemoriesController < ApplicationController
  skip_forgery_protection
  before_action :authenticate_memory_worker

  def create
    conversation_id = positive_id(params[:conversation_id])
    key = params[:run_key].to_s
    safe = AgentMemory.redact(params[:summary])
    candidate = AgentMemory.new(tenant_id: @tenant_id,
      conversation_id: conversation_id, run_key: key, summary: safe)
    unless candidate.valid?
      return render json: { error: 'invalid_memory' }, status: :unprocessable_entity
    end
    row = AgentMemory.create_or_find_by!(tenant_id: @tenant_id,
      conversation_id: conversation_id, run_key: key) do |memory|
      memory.summary = safe
    end
    render json: row.public_record
  rescue ArgumentError, TypeError
    render json: { error: 'invalid_memory' }, status: :unprocessable_entity
  end

  def index
    conversation_id = positive_id(params[:conversation_id])
    rows = AgentMemory.where(tenant_id: @tenant_id,
      conversation_id: conversation_id).order(:id).last(5)
    render json: { memories: rows.map(&:public_record) }
  rescue ArgumentError, TypeError
    render json: { error: 'invalid_memory' }, status: :unprocessable_entity
  end

  private

  def positive_id(value)
    number = Integer(value.to_s, 10)
    raise ArgumentError unless number.positive?
    number
  end

  def authenticate_memory_worker
    credentials = JSON.parse(ENV.fetch('ANCHOR_MEMORY_CREDENTIALS', '{}'))
    supplied = request.authorization.to_s.delete_prefix('Bearer ')
    match = credentials.find do |token, _tenant|
      !supplied.empty? && ActiveSupport::SecurityUtils.secure_compare(token, supplied)
    end
    @tenant_id = match && match.last
    unless @tenant_id.is_a?(Integer) && @tenant_id.positive?
      render json: { error: 'unauthorized' }, status: :unauthorized
    end
  rescue JSON::ParserError
    render json: { error: 'unauthorized' }, status: :unauthorized
  end
end
