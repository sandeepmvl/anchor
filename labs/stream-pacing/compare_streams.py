import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.streaming import AdaptivePacer, pacing_configuration

# One authored recording for the same refund ticket in both runs.
RECORDING = [(.250, 'Refunds '), (.251, 'take '), (.252, 'five '),
             (.253, 'business '), (.254, 'days.')]


def replay(options=None, disconnect=False):
    pacer = AdaptivePacer(**options) if options is not None else None
    arrivals, releases, updates = [], [], []
    displayed = ''

    def display(pieces, now):
        nonlocal displayed
        for piece in pieces:
            if piece:
                displayed += piece
                releases.extend([now] * len(piece))
                updates.append(now)

    for now, text in RECORDING:
        if pacer:
            while pacer.buffer and pacer.deadline() < now:
                deadline = pacer.deadline()
                display(pacer.due(deadline), deadline)
        arrivals.extend([now] * len(text))
        display(pacer.offer(text, now) if pacer else [text], now)
        if disconnect and pacer:
            pacer.clear()
            assert not pacer.buffer and pacer.deadline() is None
            print('Disconnected fake display: pacing buffer cleared; no final event.')
            return None
    if pacer:
        while pacer.buffer:
            deadline = pacer.deadline()
            display(pacer.due(deadline), deadline)
    assert displayed == ''.join(text for _, text in RECORDING)
    assert releases[0] == arrivals[0]
    holds = [r - a for a, r in zip(arrivals, releases)]
    if pacer:
        assert max(holds) <= pacer.delay + 1e-10
        assert pacer.peak <= pacer.maximum
    gaps = [(b - a) * 1000 for a, b in zip(updates, updates[1:])]
    ordered = sorted(gaps)
    result = {'first_byte_ms': 0.0, 'server_first_text_ms': arrivals[0] * 1000,
              'fake_render_first_text_ms': releases[0] * 1000,
              'gap_mean_ms': statistics.mean(gaps) if gaps else 0,
              'gap_p95_ms': ordered[math.ceil(.95 * len(ordered)) - 1] if gaps else 0,
              'gap_max_ms': max(gaps, default=0),
              'gap_stddev_ms': statistics.pstdev(gaps) if gaps else 0,
              'completion_ms': releases[-1] * 1000,
              'tail_ms': (releases[-1] - arrivals[-1]) * 1000,
              'max_hold_ms': max(holds) * 1000,
              'peak_buffer_chars': pacer.peak if pacer else 0}
    return result, arrivals, releases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['raw', 'paced', 'both'], default='both')
    parser.add_argument('--target', type=float)
    parser.add_argument('--chunk-chars', type=int)
    parser.add_argument('--disconnect', action='store_true')
    args = parser.parse_args()
    # Validate the deployment file, even though this comparison explicitly opts in.
    pacing_configuration()
    import yaml
    settings = yaml.safe_load(Path('config/performance.yml').read_text())['stream_pacing']
    options = {k: v for k, v in settings.items() if k != 'enabled'}
    if args.target is not None:
        options['target_interval'] = args.target
    if args.chunk_chars is not None:
        options['chunk_chars'] = args.chunk_chars
    if args.disconnect:
        replay(options, disconnect=True)
        print('Upstream cancellation is checked separately by the Python and Ruby tests.')
        return
    print('Virtual replay only: HTTP delivery, browser paint, and capacity not verified.')
    raw = replay()
    paced = replay(options)
    for name, run in [('raw', raw), ('paced', paced)]:
        if args.mode in (name, 'both'):
            print(name, json.dumps(run[0], sort_keys=True))
    if args.mode == 'both':
        _, arrivals, releases = paced
        print('character  arrival_ms  release_ms  held_ms  shared axis (0..400 ms)')
        for i in [0, 1, len(arrivals) - 2, len(arrivals) - 1]:
            a, r = arrivals[i], releases[i]
            axis = ['.'] * 81
            ai, ri = min(80, round(a / .005)), min(80, round(r / .005))
            axis[ai] = 'A'
            axis[ri] = 'X' if ai == ri else 'R'
            print(f'C{i:<3} {a*1000:10.3f} {r*1000:11.3f} {(r-a)*1000:8.3f}  ' + ''.join(axis))
        print('A=arrival/raw display R=paced release X=same 5 ms bin')
        print('Demo: The same streamed answer arrives with smoother chunk spacing '
              'while its first token is still delivered immediately.')
        print(f'Added completion tail: {paced[0]["tail_ms"]:.3f} ms; '
              'bounds take priority over perfectly even gaps.')


if __name__ == '__main__':
    main()
