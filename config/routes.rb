Rails.application.config.filter_parameters += [:summary, :authorization]

Rails.application.routes.draw do
  get "up" => "rails/health#show", as: :rails_health_check
  post "/chat", to: "chat#create"
  get "/stream_chat", to: "chat#show"
  post "/api/structured_replies", to: "api/structured_replies#create"
  get "/api/prompt_versions/active", to: "api/prompt_versions#active"
  get "/api/agent_checkpoints/:thread_id", to: "api/agent_checkpoints#show"
  post "/api/agent_checkpoints/:thread_id/approve", to: "api/agent_checkpoints#approve"
  get "/api/document_chunks", to: "api/document_chunks#index"
  post "/api/agent_memories", to: "api/agent_memories#create"
  get "/api/agent_memories", to: "api/agent_memories#index"
  post "/chat/cited", to: "chat#cited"
  post "/api/answer_feedback", to: "api/answer_feedback#create"
end
