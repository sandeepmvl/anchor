"""Replay TruLens-style feedback; no TruLens SDK or live judge is executed."""
from eval.triad import evaluate, main


def evaluate_batch(rows, client=None):
    return evaluate('trulens', rows, client)


if __name__ == '__main__':
    main('trulens')
