# Be sure to restart your server when you modify this file.

# Configure parameters to be partially matched and filtered from the log file.
# See the ActiveSupport::ParameterFilter documentation for supported notations.
Rails.application.config.filter_parameters += [
  :passw, :email, :secret, :token, :_key, :crypt, :salt, :certificate, :otp, :ssn, :cvv, :cvc
]

Rails.application.config.filter_parameters += [
  :password, :password_confirmation, :file, :bytes, :image_base64,
  :evidence, :span, :attachment_answer, :context, :cookie
]

# Whole-field filtering runs before controller-level entity detection.
Rails.application.config.filter_parameters += [
  :message, :history, :summary_state, :reply, :content, :prompt,
  :notes, :private_notes, :pii_payload
]
