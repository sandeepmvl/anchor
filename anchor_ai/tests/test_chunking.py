from dataclasses import asdict
from pathlib import Path
import re
import unittest
from anchor_ai.document_loader import load_document
from anchor_ai.chunking_lib import (
    chunk_document, FixedSizeChunker, RecursiveChunker,
    SemanticChunker, SentenceWindowChunker, embed_sentences,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'kb'
STRATEGIES = ('fixed_size', 'recursive', 'semantic', 'sentence_window',
              'hierarchical', 'structural')


def controlled_vectors(texts):
    return [[1.0, float(i % 2)] for i in range(len(texts))]


class ChunkingTests(unittest.TestCase):
    def test_all_strategies_are_repeatable_and_address_normalized_text(self):
        for name in ('policy.md', 'faq.html', 'transcript.txt'):
            document = load_document(FIXTURES / name)
            for strategy in STRATEGIES:
                with self.subTest(name=name, strategy=strategy):
                    first, scores = chunk_document(document, strategy, embed_fn=controlled_vectors)
                    second, again = chunk_document(document, strategy, embed_fn=controlled_vectors)
                    self.assertTrue(first)
                    self.assertEqual([asdict(n) for n in first], [asdict(n) for n in second])
                    self.assertEqual(scores, again)
                    self.assertEqual(len(first), len({n.id for n in first}))
                    enriched, _ = chunk_document(document, strategy, True, embed_fn=controlled_vectors)
                    for raw, contextual in zip(first, enriched):
                        self.assertTrue(0 <= raw.start < raw.end <= len(document.text))
                        self.assertEqual(raw.text, document.text[raw.start:raw.end])
                        self.assertEqual((raw.id, raw.start, raw.end, raw.text),
                                         (contextual.id, contextual.start, contextual.end, contextual.text))
                        self.assertEqual(raw.embedding_text, raw.text)
                        self.assertEqual(contextual.embedding_text,
                                         contextual.metadata['context_prefix'] + '\n' + raw.text)
                        if name == 'faq.html':
                            self.assertIsNone(re.search(r'<[^>]+>', contextual.embedding_text))
                    if strategy == 'hierarchical':
                        parents = {n.id: n for n in first if n.metadata['level'] == 'parent'}
                        children = [n for n in first if n.metadata['level'] == 'child']
                        self.assertTrue(parents and children)
                        for parent in parents.values():
                            self.assertNotIn('parent_id', parent.metadata)
                            self.assertFalse(parent.metadata['indexable'])
                        for child in children:
                            parent = parents[child.metadata['parent_id']]
                            self.assertLessEqual(parent.start, child.start)
                            self.assertLessEqual(child.end, parent.end)
                            self.assertTrue(child.metadata['indexable'])

    def test_loader_structure_and_no_html_markup(self):
        faq = load_document(FIXTURES / 'faq.html')
        self.assertEqual(faq.title, 'Acme Returns FAQ')
        self.assertEqual(len(faq.units), 2)
        self.assertNotIn('<', faq.text)
        self.assertIn('No. Opened items are excluded', faq.text)
        for name, key in [('policy.md', 'section'), ('faq.html', 'question'),
                          ('transcript.txt', 'speaker')]:
            doc = load_document(FIXTURES / name)
            nodes, _ = chunk_document(doc, 'structural')
            self.assertTrue(all(key in n.metadata for n in nodes))
        doc = load_document(FIXTURES / 'policy.md')
        nodes, _ = chunk_document(doc, 'structural')
        refund = next(n for n in nodes if n.metadata['section'] == 'Refunds')
        self.assertIn('within 30 days', refund.text)
        self.assertIn('Opened items are excluded', refund.text)
        with self.assertRaises(ValueError):
            chunk_document(doc, 'structural', options={'max_chars': 10})

    def test_window_preserves_neighbors_without_embedding_them(self):
        sample = 'Refunds allowed. Opened items excluded. Keep receipt.'
        nodes = SentenceWindowChunker(1).split(sample)
        middle = nodes[1]
        self.assertEqual(middle.metadata['window_before'], ['Refunds allowed.'])
        self.assertEqual(middle.metadata['window_after'], ['Keep receipt.'])
        self.assertEqual(middle.embedding_text, 'Opened items excluded.')
        self.assertEqual(sample[middle.metadata['window_start']:middle.metadata['window_end']], sample)
        self.assertEqual(nodes[0].metadata['window_before'], [])
        self.assertEqual(nodes[-1].metadata['window_after'], [])

    def test_size_overlap_and_coverage(self):
        text = load_document(FIXTURES / 'policy.md').text
        for cls in (FixedSizeChunker, RecursiveChunker):
            nodes = cls(80, 10).split(text)
            self.assertEqual(nodes[0].start, 0)
            self.assertEqual(nodes[-1].end, len(text))
            self.assertTrue(all(0 < len(n.text) <= 80 for n in nodes))
            for previous, current in zip(nodes, nodes[1:]):
                self.assertGreater(current.start, previous.start)
                self.assertEqual(previous.end - current.start, 10)
                self.assertEqual(current.metadata['overlap'], 10)
        self.assertEqual(RecursiveChunker(25, 0).split('Refunds allowed. Opened items excluded.')[0].text,
                         'Refunds allowed. ')

    def test_semantic_percentile_and_ties(self):
        calls = []
        def fake(texts):
            calls.append(texts)
            return [[1, 0], [1, 0], [0, 1]]
        splitter = SemanticChunker(fake, percentile=50)
        nodes = splitter.split('Refunds allowed. Keep receipt. Payment processing.')
        self.assertEqual(calls, [['Refunds allowed.', 'Keep receipt.', 'Payment processing.']])
        self.assertEqual(len(nodes), 2)
        self.assertEqual([b['split'] for b in splitter.boundaries], [False, True])
        self.assertEqual(nodes[1].start, splitter.boundaries[1]['offset'])
        tied = SemanticChunker(lambda texts: [[1, 0] for _ in texts])
        self.assertEqual(len(tied.split('One. Two. Three.')), 1)
        self.assertEqual(embed_sentences(['Refunds allowed.']), embed_sentences(['Refunds allowed.']))


if __name__ == '__main__':
    unittest.main()
