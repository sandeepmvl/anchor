from eval.dashboard import escape, render
from eval.release_gate import decide, manifest, offline_bundle


def test_failed_example_and_step_are_visible():
    expected = manifest()
    bundle = offline_bundle(expected, regressed=True)
    text = render(bundle, decide(bundle, expected, production=False))
    assert '| trajectory | failed | refund-unconfirmed | 0 |' in text
    assert '| regression | failed | refund-unconfirmed | 0 |' in text
    assert 'Tier: sample' in text and 'baseline: none' in text
    assert sum(text.count('| ' + name + ' |') for name in
               ('golden', 'calibration', 'trajectory', 'regression')) == 4


def test_untrusted_text_and_customer_pii_are_not_published():
    expected = manifest()
    bundle = offline_bundle(expected)
    report = bundle['reports']['trajectory']
    report['example_id'] = '<script>alert(1)</script> customer@example.com'
    report['answer'] = 'Customer Alice: customer@example.com, phone 202-555-0199'
    report['diagnostic'] = '| <img src=x onerror=alert(1)> |'
    text = render(bundle, decide(bundle, expected, production=False))
    assert '<script>' not in text and '<img' not in text
    assert 'customer@example.com' not in text and '202-555-0199' not in text
    assert '[redacted]' in text
    assert escape('<b>|') == '&lt;b&gt;&#124;'
