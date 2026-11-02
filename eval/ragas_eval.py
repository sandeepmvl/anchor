"""Replay RAGAS-style metrics; no RAGAS SDK or live judge is executed."""
from eval.triad import evaluate, main


def evaluate_batch(rows, client=None):
    return evaluate('ragas', rows, client)


if __name__ == '__main__':
    main('ragas')
