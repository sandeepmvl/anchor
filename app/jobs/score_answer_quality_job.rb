require 'net/http'
require 'json'
require 'uri'

class ScoreAnswerQualityJob < ApplicationJob
  queue_as :quality

  def perform(organization_id, sample_id)
    sample = AnswerQualitySample.find_by!(id: sample_id, organization_id: organization_id)
    sample.with_lock do
      return unless sample.status == 'pending'
      result = score_sample(sample)
      value = result['score']
      if result['status'] == 'insufficient'
        sample.update!(status: 'insufficient', score: nil)
      elsif result['status'] == 'scored' && value.is_a?(Numeric) &&
          value.finite? && value.between?(0, 1)
        sample.update!(status: 'scored', score: value)
      else
        raise ArgumentError, 'invalid quality result'
      end
    end
  end

  def score_sample(sample)
    secret = ENV.fetch('ANCHOR_AI_SECRET')
    raise ArgumentError, 'missing scoring secret' if secret.empty?
    uri = URI(ENV.fetch('ANCHOR_QUALITY_URL'))
    raise ArgumentError, 'invalid scoring URL' unless %w[http https].include?(uri.scheme)
    req = Net::HTTP::Post.new(uri)
    req['Content-Type'] = 'application/json'
    req['X-Anchor-Secret'] = secret
    req.body = JSON.generate(payload: sample.redacted_payload)
    timeout = AnswerQualitySample.policy.fetch('request_timeout_seconds')
    response = Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == 'https',
      open_timeout: timeout, read_timeout: timeout) { |http| http.request(req) }
    raise IOError, 'quality scoring unavailable' unless response.code == '200'
    JSON.parse(response.body)
  end
end
