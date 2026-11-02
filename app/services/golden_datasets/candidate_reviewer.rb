require 'digest'

module GoldenDatasets
  class CandidateReviewer
    def initialize(token:)
      @claims = GoldenCandidate.claims!(token, purpose: 'golden_reviewer')
    end

    def promote!(candidate_id:, corrected:)
      tenant = @claims.fetch('tenant_id').to_s
      candidate = GoldenCandidate.find_by!(id: candidate_id, tenant_id: tenant)
      draft = nil
      candidate.with_lock do
        if candidate.status == 'promoted'
          return GoldenDataset.examples.find_by!(tenant_id: tenant, key: candidate.draft_key)
        end
        raise GoldenCandidate::InvalidInput, 'queued_candidate_required' unless candidate.status == 'queued'
        raw = corrected.deep_stringify_keys
        unless raw['tenant_id'] == tenant && raw['context'].is_a?(Array) &&
            raw['context'].all? { |doc| doc.is_a?(Hash) && doc['tenant_id'] == tenant }
          raise GoldenCandidate::InvalidInput, 'wrong_tenant_context'
        end
        key = "feedback-#{candidate.id}"
        raw = raw.merge('source_run_id' => key)
        safe = GoldenDataset.sanitize(GoldenCandidate.safe!(raw))
        payload = safe.except('tenant_id', 'labeler', 'labeled_at').merge(
          'status' => 'pending', 'review_status' => 'pending', 'needs_review' => false,
          'redacted' => true, 'candidate_id' => candidate.id,
          'conversation_id' => candidate.conversation_id,
          'retrieved_chunk_ids' => candidate.retrieved_chunk_ids,
          'prompt_version' => candidate.prompt_version,
          'reviewed_by' => @claims.fetch('subject'),
          'intake_sha256' => Digest::SHA256.hexdigest(GoldenDataset.canonical(safe)))
        draft = GoldenDataset.create!(tenant_id: tenant, kind: 'example', key: key, payload: payload)
        candidate.update!(status: 'promoted', draft_key: key)
      end
      draft
    end
  end
end
