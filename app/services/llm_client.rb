class LlmClient
  CANNED_REPLY = "Anchor stub: thanks for your message, a support reply is on its way."

  def initialize(provider: Rails.configuration.x.llm.fetch(:provider))
    @provider = provider
  end

  def complete(message)
    case @provider
    when "stub"
      CANNED_REPLY
    when "openai"
      api_key
      raise NotImplementedError, "Live calls arrive in the LLM provider APIs lesson"
    else
      raise ArgumentError, "unknown provider #{@provider.inspect}"
    end
  end

  private

  def api_key
    key = ENV["OPENAI_API_KEY"]
    key = Rails.application.credentials.openai_api_key if key.nil? || key.empty?
    if key.nil? || key.empty?
      raise ArgumentError, "Set OPENAI_API_KEY before selecting the live provider"
    end
    key
  end
end
