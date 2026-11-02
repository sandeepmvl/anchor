import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anchor_ai.hybrid_retriever import HybridRuntime, RecordedHybridClient, Snapshot
from anchor_ai.rag import RecordedEvidenceClient, fixture_runtime, published_query_engine_tool


# Exercise the route and middleware in-process, without starting a server.
RAILS_CONNECTOR = r'''
require 'json'
require 'securerandom'
require 'action_dispatch/testing/integration'
ENV['ANCHOR_CHUNK_TOKENS'] = JSON.generate('local-query-engine-demo' => 1)
facts = [
  'EU customers may request a refund within 14 days.',
  'EU refund requests require an order number.',
  'Approved EU refunds return to the original payment method.'
]

# Use the existing tables, without invoking ingestion or a provider.
def insert_demo_record(model, values)
  attributes = values.stringify_keys.slice(*model.column_names)
  model.columns.each do |column|
    next if column.name == model.primary_key || attributes.key?(column.name)
    next if column.null || !column.default.nil?
    attributes[column.name] = case column.type
    when :integer, :bigint then 1
    when :float, :decimal then 0
    when :boolean then false
    when :datetime, :timestamp then Time.current
    when :date then Date.current
    when :json, :jsonb then {}
    else "query-engine-demo-#{SecureRandom.hex(8)}"
    end
  end
  result = model.insert_all!([attributes], returning: [model.primary_key])
  result.rows.first.first
end

def demo_scope
  scope = DocumentChunk.where(document_id: Document.where(tenant_id: 1).select(:id))
  if DocumentChunk.column_names.include?('tenant_id')
    scope = scope.where(tenant_id: 1)
  end
  scope
end

if ENV['ANCHOR_DEMO_SEED'] == '1'
  Document.transaction do
    missing = facts - demo_scope.where(text: facts).pluck(:text)
    unless missing.empty?
      document_id = insert_demo_record(Document, tenant_id: 1, title: 'Query engine EU policy fixture')
      missing.each_with_index do |text, chunk_index|
        insert_demo_record(DocumentChunk, document_id: document_id, tenant_id: 1,
          chunk_index: chunk_index, text: text)
      end
    end
  end
end
ids = JSON.parse(ENV.fetch('ANCHOR_DEMO_IDS', '[]'))
if ids.empty?
  by_text = demo_scope.where(text: facts).order(:id).to_a.group_by(&:text)
  ids = facts.filter_map { |text| by_text[text]&.first&.id }
end
raise 'Run this demo with --seed first; three EU policy rows are required' unless ids.length == 3
session = ActionDispatch::Integration::Session.new(Rails.application)
session.host! 'localhost'
session.get '/api/document_chunks', params: { ids: ids.map(&:to_s) }, headers: {
  'Authorization' => 'Bearer local-query-engine-demo',
  'Accept' => 'application/json',
  'User-Agent' => 'Mozilla/5.0 Chrome/131.0.0.0 Safari/537.36'
}
response = session.response
unless response.status == 200
  raise "Hydration failed: #{response.status}: #{response.body}"
end
puts 'ANCHOR_CONNECTOR_JSON=' + response.body
'''


def hydrate(ids=None, *, seed=False):
    env = dict(os.environ, ANCHOR_LLM_PROVIDER='offline',
               ANCHOR_DEMO_IDS=json.dumps(ids or []),
               ANCHOR_DEMO_SEED='1' if seed else '0')
    result = subprocess.run(['bin/rails', 'runner', '-'], cwd=ROOT,
                            input=RAILS_CONNECTOR, env=env, text=True,
                            capture_output=True, timeout=8)
    if result.returncode:
        raise RuntimeError('Rails connector failed:\n' + result.stdout + result.stderr)
    for line in reversed(result.stdout.splitlines()):
        if line.startswith('ANCHOR_CONNECTOR_JSON='):
            return json.loads(line.split('=', 1)[1])
    raise RuntimeError('Rails connector returned no payload:\n' + result.stdout + result.stderr)


class DatabaseRecordedClient(RecordedHybridClient):
    """Authored rankings over hydrated rows; not a relevance benchmark."""

    def __init__(self, payload):
        self.snapshot = Snapshot(payload['tenant_id'], 'offline-db-snapshot-v1', tuple(
            dict(id=row['id'], tenant_id=payload['tenant_id'], text=row['text'])
            for row in payload['chunks']))

    def search(self, branch, query, snapshot, depth):
        # Return selected excerpts through both existing retrieval branches.
        return [dict(id=row['id'], tenant_id=snapshot.tenant_id,
                     generation_id=snapshot.generation_id)
                for row in snapshot.chunks[:depth]]


async def compare(query):
    payload = hydrate()
    client = DatabaseRecordedClient(payload)
    runtime = HybridRuntime(payload['tenant_id'], client.acquire_snapshot,
                            client.dense_search, client.sparse_search)
    generation = RecordedEvidenceClient()
    tool = published_query_engine_tool(runtime, payload['tenant_id'], client=generation)
    result = await tool(query)
    assert not result['abstained'], result
    assert len(result['source_nodes']) == 3, result
    assert len(result['citations']) == 3, result
    opened = hydrate(result['citations'])
    database = {row['id']: row for row in opened['chunks']}
    for node in result['source_nodes']:
        assert database[node['document_chunk_id']]['text'] == node['text']
    # This independent baseline has recordings for its own question only.
    # It compares response contracts, not relevance on a shared corpus.
    dense_query = 'How do I get a refund?'
    dense = fixture_runtime(tenant_id=1).retriever
    dense_hits = dense.search(dense_query, 3)
    print(json.dumps(dict(
        raw_dense_service=dict(query=dense_query, chunks=dense_hits),
        query=query,
        query_engine=result,
        hydrated_rows=opened,
        comparison='Dense returns chunks; the query engine adds synthesis and citations.',
        verification='Three citations resolve to authorized document_chunks rows.',
        retrieval_generation_note=(
            'offline-db-snapshot-v1 identifies this demo snapshot; stored generation '
            'fields, when present, are in hydrated_rows.chunks.retrieval_metadata.'),
    ), indent=2, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', default='refund policy for EU customers')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--seed', action='store_true')
    args = parser.parse_args()
    if args.seed:
        print(json.dumps(hydrate(seed=True), indent=2))
    elif args.inspect:
        print(json.dumps(hydrate(), indent=2))
    else:
        asyncio.run(compare(args.query))


if __name__ == '__main__':
    main()
