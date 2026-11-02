require 'openssl'
require 'securerandom'

class User < ApplicationRecord
  has_many :memberships
  has_many :tickets
  validates :email, :password_digest, presence: true
  validates :email, uniqueness: true
  ITERATIONS = 210_000

  def password=(value)
    raise ArgumentError, 'password_length' unless value.is_a?(String) && value.bytesize.between?(12, 128)
    salt = SecureRandom.hex(16)
    digest = OpenSSL::PKCS5.pbkdf2_hmac(value, salt, ITERATIONS, 32, 'sha256').unpack1('H*')
    self.password_digest = [salt, digest].join(':')
  end

  def authenticate(value)
    return false unless value.is_a?(String) && value.bytesize.between?(12, 128)
    salt, expected = password_digest.to_s.split(':', 2)
    return false unless salt&.match?(/\A[0-9a-f]{32}\z/) && expected&.match?(/\A[0-9a-f]{64}\z/)
    actual = OpenSSL::PKCS5.pbkdf2_hmac(value, salt, ITERATIONS, 32, 'sha256').unpack1('H*')
    ActiveSupport::SecurityUtils.secure_compare(actual, expected) && self
  end
end
