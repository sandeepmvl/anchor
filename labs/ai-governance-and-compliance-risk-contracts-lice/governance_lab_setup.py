from pathlib import Path
import json

root = Path.cwd()
if not (root / 'bin/rails').exists():
    raise SystemExit('Run from the Anchor Rails project root')

inventory = [
    {'name': 'base_model', 'version': 'fixture-v1', 'licence': 'Fictional commercial terms',
     'source': 'docs/governance/fixtures/assets.json#BM-1',
     'permitted_uses': ['commercial_saas', 'fine_tuning'], 'provenance_review': 'documented'},
    {'name': 'embedding_model', 'version': 'fixture-v1', 'licence': 'Fictional non-commercial terms',
     'source': 'docs/governance/fixtures/assets.json#EM-1',
     'permitted_uses': ['research'], 'provenance_review': 'documented'},
    {'name': 'eval_dataset', 'version': 'fixture-v1', 'licence': 'CC0-1.0 claimed',
     'source': 'docs/governance/fixtures/assets.json#ED-1',
     'permitted_uses': ['commercial_saas'], 'provenance_review': 'unresolved'}
]
assets = {
    'BM-1': {'terms': 'FICTIONAL: commercial hosted SaaS and dedicated fine-tuning allowed; retain attribution. Redistribution not approved.', 'origin': 'Created fixture; no third-party model downloaded.'},
    'EM-1': {'terms': 'FICTIONAL: research only; commercial use prohibited.', 'origin': 'Created fixture; no embedding weights downloaded.'},
    'ED-1': {'terms': 'FICTIONAL uploader claims CC0; upstream collection rights undocumented.', 'origin': 'Unknown upstream rights in this fictional review.'}
}
files = {
'config/licence_inventory.yml': json.dumps(inventory, indent=2) + '\n',
'docs/governance/fixtures/assets.json': json.dumps(assets, indent=2) + '\n',
'docs/governance/fixtures/provider_agreement.md': '''# Fictional provider agreement — not actual provider terms
1. Service: process prompts and completions only on documented instructions to generate responses.
2. Improvement: provider may retain prompts and completions to improve general models for its own purposes.
3. Logs: customer-configured diagnostic logs retained for 30 days solely for that customer's troubleshooting.
4. Fine-tuning: tenant training files used only for that tenant's dedicated model on instructions.
5. Other: sub-processors may be used; improvement retention is determined by provider.
Missing: authorization/change procedure, assistance, deletion/return, audit commitments and transfer details.
''',
'scripts/licence_check.py': '''import argparse
import json
from pathlib import Path


def check(entries, declared_use):
    failures = []
    for entry in entries:
        reasons = []
        for field in ('name', 'licence', 'source', 'version'):
            if not entry.get(field):
                reasons.append('missing ' + field)
        if declared_use not in entry.get('permitted_uses', []):
            reasons.append('permission missing: ' + declared_use)
        if entry.get('provenance_review') != 'documented':
            reasons.append('provenance unresolved')
        if reasons:
            failures.append({'asset': entry.get('name', '<unnamed>'), 'reasons': reasons})
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', default='config/licence_inventory.yml')
    parser.add_argument('--use', default='commercial_saas')
    args = parser.parse_args()
    try:
        entries = json.loads(Path(args.inventory).read_text(encoding='utf-8'))
        failures = check(entries, args.use)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print(json.dumps({'status': 'invalid_inventory', 'error': str(error)}))
        return 2
    print(json.dumps({'status': 'blocked' if failures else 'metadata_pass', 'failures': failures}, sort_keys=True))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
''',
'db/migrate/20260919000100_create_governance_decisions.rb': '''class CreateGovernanceDecisions < ActiveRecord::Migration[8.1]
  def change
    create_table :governance_decisions do |t|
      t.string :tenant_ref, null: false
      t.string :actor_ref, null: false
      t.string :decision, null: false
      t.string :status, null: false
      t.json :basis_refs, null: false, default: []
      t.string :artifact_version, null: false
      t.timestamps
    end
    add_index :governance_decisions, [:tenant_ref, :created_at]
  end
end
''',
'app/models/governance_decision.rb': '''class GovernanceDecision < ApplicationRecord
  scope :for_tenant, ->(tenant_ref) { where(tenant_ref: tenant_ref) }
  validates :tenant_ref, :actor_ref, :decision, :status, :artifact_version, presence: true
  validate :controlled_basis_references

  private

  def controlled_basis_references
    unless basis_refs.is_a?(Array) && basis_refs.any? &&
        basis_refs.all? { |ref| ref.is_a?(String) && ref.match?(/\\A[A-Z]{2,8}-[0-9]+\\z/) }
      errors.add(:basis_refs, 'must be controlled reference IDs, not customer content')
    end
  end
end
''',
'app/serializers/governance_decision_serializer.rb': '''class GovernanceDecisionSerializer
  def self.call(record, tenant_ref:)
    raise SecurityError, 'tenant mismatch' unless record.tenant_ref == tenant_ref
    {
      event_type: 'governance_decision',
      tenant_ref: record.tenant_ref,
      actor_ref: record.actor_ref,
      decision: record.decision,
      status: record.status,
      basis_refs: record.basis_refs,
      artifact_version: record.artifact_version,
      timestamp: record.created_at.iso8601
    }
  end
end
''',
'docs/governance/open_questions.md': '''# Retail EU governance register — fictional exercise
| ID | Evidence / issue | Owner | Release consequence / next action |
|---|---|---|---|
| RC-7 | Delivery-policy explanations only; current legal applicability unverified | Counsel and governance reviewer | Hold legal sign-off pending dated source review; reopen on purpose change |
| DPA-2 | Agreement clause 2 permits provider own-purpose model training | Counsel and procurement | Hold provider approval; assess lawful basis, disclosure and authority; disable unapproved reuse where possible |
| DPA-5 | Clause 5 lacks authorization/change, assistance, deletion/return and audit terms | Procurement and counsel | Hold agreement approval; obtain sufficient processing and transfer terms |
| EM-1 | Research-only permission excludes commercial SaaS | Asset owner | Block asset; obtain permission or replace it |
| ED-1 | Upstream collection rights undocumented | Dataset owner and counsel | Hold dataset; request lineage and rights evidence |

Scope: no credit, employment, biometric, essential-service eligibility or regulated-product safety decisions.
Evidence references belong in access-controlled storage, not customer-message logs.
Actor references can still be personal data; production retention and access controls remain required.
Passing the local metadata gate does not close RC-7, DPA-2 or DPA-5.
'''
}
for name, content in files.items():
    path = root / name
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise SystemExit('Refusing to overwrite: ' + name)
for name, content in files.items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
print('CREATED governance lab files')
