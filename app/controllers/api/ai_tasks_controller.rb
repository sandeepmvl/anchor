module Api
  class AiTasksController < ApplicationController
    before_action :authorize_organization!

    def create
      # The upload boundary must already have authorized these tenant-owned
      # references. The ingestion service also enforces reference ownership.
      references = params.require(:references)
      raise ArgumentError unless references.is_a?(Array)
      task = AiTaskDispatcher.submit_ingestion!(organization_id: @organization_id,
        submission_key: request.headers["Idempotency-Key"].to_s,
        references: references)
      location = "/api/organizations/#{@organization_id}/ai_tasks/#{task.id}"
      response.set_header("Location", location)
      response.set_header("Retry-After", "2")
      render json: { id: task.id, state: task.state, status_url: location }, status: :accepted
    rescue AiTask::Conflict
      render json: { error: "submission_key_conflict" }, status: :conflict
    rescue ArgumentError, ActionController::ParameterMissing
      render json: { error: "invalid_task" }, status: :unprocessable_entity
    rescue AiTaskDispatcher::ReleaseBlocked
      render json: { error: "task_service_not_ready" }, status: :service_unavailable
    end

    def show
      task = AiTask.find_by(id: params[:id], organization_id: @organization_id)
      return head :not_found unless task
      response.set_header("Cache-Control", "no-store")
      render json: task.public_status
    end

    private

    def authorize_organization!
      user_id = session[:user_id]
      return head :unauthorized unless user_id
      organization_id = Integer(params[:organization_id], exception: false)
      membership = Membership.find_by(user_id: user_id, organization_id: organization_id)
      return head :not_found unless membership
      @organization_id = membership.organization_id
    end
  end
end
