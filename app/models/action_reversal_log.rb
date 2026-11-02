class ActionReversalLog < ApplicationRecord
  belongs_to :tool_action
  validates :event, inclusion: { in: %w[execution compensation] }
  before_update { raise ActiveRecord::ReadOnlyRecord, 'audit events are append-only' }
  before_destroy { raise ActiveRecord::ReadOnlyRecord, 'audit events are append-only' }
end
