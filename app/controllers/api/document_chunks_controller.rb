module Api
  class DocumentChunksController < ApplicationController
    def index
      tenant_id = service_tenant
      return head :unauthorized unless tenant_id

      raw = params[:ids]
      unless raw.is_a?(Array) && raw.length.between?(1, 20) &&
          raw.all? { |id| id.is_a?(String) && id.match?(/\A[1-9][0-9]{0,17}\z/) }
        return render json: { error: 'invalid_chunk_ids' }, status: :bad_request
      end
      ids = raw.map(&:to_i)
      return head :bad_request unless ids.uniq.length == ids.length

      rows = authorized_chunks(tenant_id).where(id: ids).to_a
      return head :not_found unless rows.length == ids.length

      by_id = rows.index_by(&:id)
      render json: { tenant_id: tenant_id, chunks: ids.map { |id| serialize_chunk(by_id.fetch(id)) } }
    end

    private

    def service_tenant
      token = request.authorization.to_s.delete_prefix('Bearer ')
      return nil if token.empty?
      tokens = JSON.parse(ENV.fetch('ANCHOR_CHUNK_TOKENS', '{}'))
      return nil unless tokens.is_a?(Hash)
      pair = tokens.find do |secret, tenant|
        secret.is_a?(String) && !secret.empty? && tenant.is_a?(Integer) && tenant.positive? &&
          ActiveSupport::SecurityUtils.secure_compare(secret, token)
      end
      pair&.last
    rescue JSON::ParserError
      nil
    end

    def authorized_chunks(tenant_id)
      # Ownership must be established on the document, not inferred from an ID.
      documents = Document.where(tenant_id: tenant_id).select(:id)
      scope = DocumentChunk.where(document_id: documents)
      if DocumentChunk.column_names.include?('tenant_id')
        scope = scope.where(tenant_id: tenant_id)
      end
      scope
    end

    def serialize_chunk(row)
      attributes = row.attributes
      metadata = attributes.select do |key, _|
        key.include?('generation') || %w[embedding_model embedding_version chunking_version].include?(key)
      end
      {
        id: row.id,
        document_id: row.document_id,
        text: attributes.fetch('text'),
        retrieval_metadata: metadata
      }
    end
  end
end
