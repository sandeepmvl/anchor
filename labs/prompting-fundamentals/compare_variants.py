import json
from pathlib import Path
import sys

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anchor_ai.llm_client import LlmClient
from anchor_ai.prompts import VARIANTS


def main():
    tickets = json.loads(
        (ROOT / 'anchor_ai/tests/fixtures/triage_tickets.json').read_text()
    )
    print('Same support ticket, three named prompt strategies — offline results')
    print('Synthetic responses; these scores are not model-quality evidence.')
    print('variant | category | urgency | both | valid | paraphrase pair agrees')
    for variant in VARIANTS:
        client = LlmClient(provider='offline')
        predictions = {}
        category = urgency = both = valid = 0
        for ticket in tickets:
            try:
                result = client.complete_triage(
                    ticket['ticket'], variant=variant, fixture_id=ticket['id']
                ).model_dump()
            except ValidationError:
                predictions[ticket['id']] = None
                continue
            predictions[ticket['id']] = result
            valid += 1
            category_ok = result['category'] == ticket['category']
            urgency_ok = result['urgency'] == ticket['urgency']
            category += category_ok
            urgency += urgency_ok
            both += category_ok and urgency_ok
        stable = (predictions['t4'] is not None
                  and predictions['t4'] == predictions['t5'])
        count = len(tickets)
        print(f'{variant} | {category}/{count} | {urgency}/{count} | '
              f'{both}/{count} | {valid}/{count} | {"Yes" if stable else "No"}')
        print(f'  t3: {json.dumps(predictions["t3"], sort_keys=True)}')


if __name__ == '__main__':
    main()
