require_relative "boot"
require "rails/all"

Bundler.require(*Rails.groups)

module Anchor
  class Application < Rails::Application
    config.load_defaults 8.1
    config.active_record.schema_format = :sql
    config.autoload_lib(ignore: %w[assets tasks])
    config.x.llm = config_for(:llm)
  end
end
