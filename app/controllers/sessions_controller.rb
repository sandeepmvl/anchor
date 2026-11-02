class SessionsController < ApplicationController
  protect_from_forgery with: :exception

  def create
    user = User.find_by(email: params[:email].to_s.downcase)
    member = user&.memberships&.find_by(organization_id: params[:organization_id])
    unless user&.authenticate(params[:password]) && member
      return head :unauthorized
    end
    reset_session
    session[:user_id] = user.id
    session[:membership_id] = member.id
    head :no_content
  end

  def destroy
    reset_session
    head :no_content
  end
end
