# Required application data; safe to execute repeatedly.
# Existing published contracts must match rather than being overwritten.
{
  'chat:v1' => PromptRegistry::V1,
  'chat:v2' => PromptRegistry::V2
}.each do |version, attributes|
  PromptRegistry.publish(version, attributes)
end
