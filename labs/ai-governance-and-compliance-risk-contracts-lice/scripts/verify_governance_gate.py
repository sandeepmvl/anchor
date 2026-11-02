from pathlib import Path
import json
import subprocess
import sys


def run_gate(path):
    result = subprocess.run(
        [sys.executable, 'scripts/licence_check.py', '--inventory', str(path)],
        capture_output=True, text=True, timeout=10, check=False
    )
    return result.returncode, json.loads(result.stdout)


original = Path('config/licence_inventory.yml')
code, result = run_gate(original)
assert code == 1, (code, result)
assert [item['asset'] for item in result['failures']] == ['embedding_model', 'eval_dataset']
assert result['failures'][0]['reasons'] == ['permission missing: commercial_saas']
assert result['failures'][1]['reasons'] == ['provenance unresolved']
print('PASS initial blockers: embedding_model, eval_dataset')

entries = json.loads(original.read_text(encoding='utf-8'))
evidence = {
    'EM-2': 'FICTIONAL replacement terms expressly allow commercial_saas; created fixture origin documented.',
    'ED-2': 'FICTIONAL replacement evaluation dataset authored entirely for this lab; commercial evaluation expressly permitted.'
}
Path('docs/governance/fixtures/remediation.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
entries[1].update(version='fixture-v2', licence='Fictional commercial replacement',
                  source='docs/governance/fixtures/remediation.json#EM-2',
                  permitted_uses=['commercial_saas'])
entries[2].update(version='fixture-v2', licence='Fictional authored evaluation fixture',
                  source='docs/governance/fixtures/remediation.json#ED-2',
                  provenance_review='documented')
remediated = Path('config/licence_inventory.remediated.yml')
remediated.write_text(json.dumps(entries, indent=2), encoding='utf-8')
code, result = run_gate(remediated)
assert code == 0 and result == {'status': 'metadata_pass', 'failures': []}, (code, result)
assert json.loads(original.read_text(encoding='utf-8'))[2]['provenance_review'] == 'unresolved'
print('PASS remediated metadata only; contract and classification holds remain')
