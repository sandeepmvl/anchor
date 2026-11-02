import os


class LlmClient:
    def __init__(self, provider=None):
        self.provider = provider or os.environ.get("ANCHOR_AI_PROVIDER", "offline")

    def complete(self, prompt, system=None, max_tokens=256):
        if self.provider != "offline":
            raise NotImplementedError("Live providers arrive in a later lesson")
        return "Offline support reply: please check the tenant's return policy."
