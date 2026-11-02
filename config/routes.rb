Rails.application.config.filter_parameters += [:summary, :authorization]
Rails.application.config.filter_parameters += [:credential]

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
  post '/session', to: 'sessions#create'
  delete '/session', to: 'sessions#destroy'
  get '/tickets/:ticket_id', to: 'api/ticket_attachments#index', as: :attachment_ticket
  get '/api/tickets/:ticket_id/attachments/:id', to: 'api/ticket_attachments#show', as: :ticket_attachment_evidence
  post '/api/tickets/:ticket_id/attachments', to: 'api/ticket_attachments#create', as: :ticket_attachments
  post '/api/realtime_sessions', to: 'api/realtime_sessions#create'
  post '/api/organizations/:organization_id/ai_tasks', to: 'api/ai_tasks#create'
  get '/api/organizations/:organization_id/ai_tasks/:id', to: 'api/ai_tasks#show'
  post '/api/ai_costs', to: 'api/ai_costs#create'
  get '/admin/organizations/:organization_id/ai_costs', to: 'admin/ai_costs#index'
end
