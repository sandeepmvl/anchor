import argparse
import asyncio
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.dataset_quality import read_rows
from anchor_ai.distillation import evaluate
from anchor_ai.query_analyzer import QueryAnalyzer


class OfflineQueryClient:
    def __init__(self):
        self.calls = []

    async def complete(self, ticket):
        self.calls.append(ticket)
        return json.dumps({'queries': [ticket]})


async def demo(student):
    client = OfflineQueryClient()
    analyzer = QueryAnalyzer(client, classifier=student)
    results = []
    for text, task in [('Please refund chargeback.', 'intent'),
                       ('Please refund chargeback.', 'tone'),
                       ('quasar nebula zephyr', 'intent'),
                       ('Can I get a refund under the policy?', 'policy_answering')]:
        result = await analyzer.analyze_with_classification(text, task, 'demo')
        results.append({'ticket': text, **result})
    return {'cases': results, 'existing_analyzer_calls': len(client.calls)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    report, student = evaluate(read_rows(args.predictions), args.balanced)
    report['environment'] = {'python': sys.version, 'platform': platform.platform()}
    if args.demo:
        report['demo'] = asyncio.run(demo(student))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
