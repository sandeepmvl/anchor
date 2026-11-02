import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anchor_ai.document_loader import load_document
from anchor_ai.chunking_lib import chunk_document


def main():
    config = yaml.safe_load((ROOT / 'config/retrieval.yml').read_text())['chunking']
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc', required=True)
    parser.add_argument('--strategy', choices=list(config['strategies']),
                        default=config['default_strategy'])
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--normalized', action='store_true')
    args = parser.parse_args()
    document = load_document(args.doc)
    if args.normalized:
        print(f'Normalized source: {document.source}')
        print(document.text)
        print(json.dumps([asdict(u) for u in document.units], ensure_ascii=False))
        return
    nodes, boundaries = chunk_document(
        document, args.strategy,
        context=args.context or config['context_prefix'],
        options=config['strategies'][args.strategy])
    print(json.dumps(dict(doc=document.source, strategy=args.strategy,
                          node_count=len(nodes),
                          searchable_nodes=sum(n.metadata['indexable'] for n in nodes),
                          size_unit='characters; window radius is sentences')))
    for item in nodes:
        assert document.text[item.start:item.end] == item.text
        record = asdict(item)
        record['embedding_text'] = item.embedding_text
        print(json.dumps(record, ensure_ascii=False))
    if args.strategy == 'sentence_window':
        refund = next((n for n in nodes if 'Refunds are available within 30 days.' in n.text), None)
        if refund:
            print('Anchor demo: searchable refund sentence with surrounding context:')
            print(document.text[refund.metadata['window_start']:refund.metadata['window_end']])
    if boundaries:
        print(json.dumps({'candidate_breakpoints': boundaries}, indent=2))


if __name__ == '__main__':
    main()
