if defined?(Rails::Application) && Rails.application
  require "rbconfig"

  class AiTaskFlowIsolationTest < ActiveSupport::TestCase
    test "durable task acceptance exercise passes in an isolated process" do
      assert system(RbConfig.ruby, File.expand_path(__FILE__)),
        "Standalone task acceptance tests failed"
    end
  end
else
  require_relative "../../labs/durable-background-jobs/test_support"

  class AiTaskFlowTest < ActionController::TestCase
    tests Api::AiTasksController

    def setup
      DurableTaskLab.reset!
      Membership.create!(user_id: 11, organization_id: 1)
      Membership.create!(user_id: 22, organization_id: 2)
      @routes = ActionDispatch::Routing::RouteSet.new
      @routes.draw do
        post "/api/organizations/:organization_id/ai_tasks", to: "api/ai_tasks#create"
        get "/api/organizations/:organization_id/ai_tasks/:id", to: "api/ai_tasks#show"
      end
      @request.session[:user_id] = 11
      @request.headers["Idempotency-Key"] = "large-policy"
    end

    def test_acceptance_and_tenant_scoped_polling
      references = Array.new(500) { |i| "%032x" % i }
      post :create, params: { organization_id: 1, references: references }
      assert_response :accepted
      receipt = JSON.parse(@response.body)
      assert_equal "queued", receipt.fetch("state")
      assert_equal "/api/organizations/1/ai_tasks/#{receipt.fetch('id')}", @response.headers["Location"]
      assert_equal 0, AiTask.find(receipt.fetch("id")).attempts
      get :show, params: { organization_id: 1, id: receipt.fetch("id") }
      assert_response :success
      assert_equal 0, JSON.parse(@response.body).fetch("progress")
      @request.session[:user_id] = 22
      get :show, params: { organization_id: 2, id: receipt.fetch("id") }
      assert_response :not_found
      get :show, params: { organization_id: 1, id: receipt.fetch("id") }
      assert_response :not_found
      puts "large upload: immediate 202 with tenant-scoped status URL; cross-tenant polling denied"
    end

    def test_lost_acceptance_response_reuses_receipt
      args = { organization_id: 1, references: ["a" * 32] }
      2.times do
        post :create, params: args
        assert_response :accepted
      end
      assert_equal 1, AiTask.count
      post :create, params: { organization_id: 1, references: ["b" * 32] }
      assert_response :conflict
    end

    def test_missing_session_is_rejected
      @request.session.delete(:user_id)
      post :create, params: { organization_id: 1, references: ["a" * 32] }
      assert_response :unauthorized
      assert_equal 0, AiTask.count
    end

    def test_status_does_not_expose_private_payload
      task = DurableTaskLab.submit
      get :show, params: { organization_id: 1, id: task.id }
      assert_response :success
      body = JSON.parse(@response.body)
      refute body.key?("payload_json")
      refute body.key?("claim_token")
      assert_equal "no-store", @response.headers["Cache-Control"]
    end
  end
end
