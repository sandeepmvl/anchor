class Api::RealtimeSessionsController < ApplicationController
  def create
    response.headers['Cache-Control'] = 'no-store'
    result = MultimodalClient.new.realtime_session(
      session: session, ticket_id: params.require(:ticket_id),
      origin: request.headers['Origin'])
    AiTelemetry.realtime_issuance(status: 'issued')
    render json: result, status: :created
  rescue RealtimeSessionIssuer::Denied
    AiTelemetry.realtime_issuance(status: 'denied')
    render json: { error: 'realtime_not_authorized' }, status: :forbidden
  end
end
