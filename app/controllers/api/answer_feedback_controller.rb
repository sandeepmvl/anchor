class Api::AnswerFeedbackController < ApplicationController
  protect_from_forgery with: :exception

  def create
    claims = GoldenCandidate.claims!(customer_token, purpose: 'answer_customer')
    raise GoldenCandidate::InvalidInput, 'down_required' unless params[:rating] == 'down'
    feedback, candidate = GoldenCandidate.feedback!(claims: claims,
      message_id: params.require(:message_id), comment: params[:comment])
    render json: {feedback_event_id: feedback.id, golden_candidate_id: candidate.id,
      status: candidate.status}
  rescue GoldenCandidate::AuthorizationError
    head :unauthorized
  rescue ActiveRecord::RecordNotFound
    head :not_found
  rescue GoldenCandidate::InvalidInput
    head :unprocessable_entity
  end

  private

  def customer_token
    session[:anchor_customer_context].presence || request.authorization.to_s.delete_prefix('Bearer ')
  end
end
