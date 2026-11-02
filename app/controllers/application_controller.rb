class ApplicationController < ActionController::Base
  # Only allow modern browsers supporting webp images, web push, badges, import maps, CSS nesting, and CSS :has.
  allow_browser versions: :modern

  # Changes to the importmap will invalidate the etag for HTML responses
  stale_when_importmap_changes

  # Provenance is diagnostic metadata, never an authorization credential.
  after_action :attach_anchor_release_provenance

  private

  def attach_anchor_release_provenance
    release = ENV['ANCHOR_RELEASE_ID'].to_s
    if release.match?(/\A[a-f0-9]{40,64}\z/)
      response.set_header('X-Anchor-Release', release)
    end
  end
end
