Rails.application.routes.draw do
  get "up" => "rails/health#show", as: :rails_health_check
  post "/chat", to: "chat#create"
  post "/api/structured_replies", to: "api/structured_replies#create"
end
