#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec ruby labs/incident-response-playbooks/rehearsal.rb "${1:-simulate}"
