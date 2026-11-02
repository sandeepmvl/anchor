import hashlib
import math
from pathlib import Path

import yaml
from eval.groundedness_scorer import score_online


def policy():
    p = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/online_quality.yml').read_text())
    if not 0 <= p['sample_rate'] <= 1:
        raise ValueError('invalid_sample_rate')
    for key in ('minimum_valid', 'sustained_windows', 'window_seconds', 'daily_tenant_limit'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError('invalid_' + key)
    if not 0 <= p['floor'] <= 1:
        raise ValueError('invalid_floor')
    return p


def selected(identity, p):
    digest = hashlib.sha256((p['seed'] + ':' + identity).encode()).hexdigest()
    return int(digest[:13], 16) / 16**13 < p['sample_rate']


def score(payload, judge=None):
    result = score_online(payload, judge=judge)
    value = result.get('score')
    valid = (result.get('status') == 'scored' and type(value) in (int, float)
             and math.isfinite(value) and 0 <= value <= 1)
    return {'status': 'scored' if valid else 'insufficient',
            'score': value if valid else None}


def windows(rows, p):
    """Rows represent CLOSED windows, including explicit empty windows.

    Each row has group, window, and scores. The caller partitions by event time
    and closes windows once; retries must reuse the same immutable inputs.
    """
    streaks, previous, output, seen = {}, {}, [], set()
    for row in sorted(rows, key=lambda r: (tuple(r['group']), r['window'])):
        group = tuple(row['group'])
        window = row['window']
        identity = (group, window)
        if identity in seen:
            raise ValueError('duplicate_closed_window')
        seen.add(identity)
        values = [v for v in row['scores'] if type(v) in (int, float)
                  and math.isfinite(v) and 0 <= v <= 1]
        if previous.get(group) != window - 1:
            streaks[group] = 0
        previous[group] = window
        enough = len(values) >= p['minimum_valid']
        mean = sum(values) / len(values) if enough else None
        bad = enough and mean < p['floor']
        streaks[group] = streaks.get(group, 0) + 1 if bad else 0
        output.append({'group': list(group), 'window': window,
                       'samples': len(values), 'mean_groundedness': mean,
                       'status': 'scored' if enough else 'insufficient',
                       'drift': streaks[group] >= p['sustained_windows']})
    return output


if __name__ == '__main__':
    import json
    p = policy()
    rows = [{'group': ['cedar', 'refund', 'provider-v2', p['evaluator_version']],
             'window': w, 'scores': values}
            for w, values in enumerate(([1.0]*5, [0.0]*5, [0.0]*5))]
    for item in windows(rows, p):
        print(json.dumps(item))
    print('Same ungrounded answer contributes to quality trend and sustained drift alert; customer response does not wait for judging.')
    print('Offline smoke: provider quality, queue delivery, Grafana and notification delivery not verified.')
