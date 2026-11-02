import html
import re

CHECKS = ('golden', 'calibration', 'trajectory', 'regression')


def escape(value):
    return html.escape(str(value), quote=True).replace('|', '&#124;').replace('\n', ' ')


def identifier(value):
    # Only synthetic dataset identifiers, never free text or customer identifiers.
    if isinstance(value, str) and re.fullmatch(r'(refund|case|golden|regression)-[a-z0-9-]{1,48}', value):
        return escape(value)
    return '[redacted]'


def render(bundle, verdict):
    state = 'PASS' if verdict['passed'] else 'FAIL'
    baseline = bundle.get('baseline', {}).get('source', {}).get('run_id')
    baseline = str(baseline) if baseline is not None else 'none'
    if not re.fullmatch(r'[0-9]+|none', baseline):
        baseline = '[redacted]'
    tier = bundle.get('tier') if bundle.get('tier') in ('sample', 'full') else 'invalid'
    scope = bundle.get('scope') if bundle.get('scope') in ('offline', 'production') else 'invalid'
    text = f'## Anchor evaluation: {state}\n\nTier: {tier}; scope: {scope}; baseline: {baseline}\n\n'
    text += '| Evaluator | State | Example | Trajectory step |\n|---|---|---|---|\n'
    for name in CHECKS:
        report = bundle.get('reports', {}).get(name, {})
        step = report.get('step')
        step = str(step) if type(step) is int and 0 <= step < 100000 else 'unknown'
        row_state = verdict['rows'].get(name, 'failed')
        if row_state not in ('passed', 'failed', 'quarantined'):
            row_state = 'failed'
        text += f'| {name} | {row_state} | {identifier(report.get("example_id"))} | {step} |\n'
    text += '\nManifest SHA-256: ' + escape(__import__('eval.release_gate', fromlist=['digest']).digest(bundle.get('manifest'))) + '\n'
    text += '\nFailure codes: '
    # Failure codes originate in validator code, not evaluator prose.
    text += ', '.join(escape(code) for code in verdict['errors']) or 'none'
    text += '\n\nCustomer text and raw tool arguments are excluded.\n'
    return text
