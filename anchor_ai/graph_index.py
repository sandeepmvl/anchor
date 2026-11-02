import json
from .index_maintenance import canonical, digest, valid_tenant, active

TYPES = {'product', 'plan', 'policy', 'exception'}
RELATIONS = {'on_plan': ('product', 'plan'),
             'uses_policy': ('plan', 'policy'),
             'grants_exception': ('policy', 'exception'),
             'includes_product': ('plan', 'product')}


def ensure_graph(db):
    # execute(), not executescript(): never implicitly commit the caller's transaction.
    for sql in (
        '''CREATE TABLE IF NOT EXISTS graph_sources(
           document_id INTEGER,generation_id INTEGER,signature TEXT,
           PRIMARY KEY(document_id,generation_id))''',
        '''CREATE TABLE IF NOT EXISTS graph_nodes(
           tenant_id INTEGER,scope TEXT,generation_id INTEGER,document_id INTEGER,
           node_id TEXT,kind TEXT,aliases TEXT,source_hash TEXT)''',
        '''CREATE TABLE IF NOT EXISTS graph_edges(
           tenant_id INTEGER,scope TEXT,generation_id INTEGER,document_id INTEGER,
           source TEXT,relation TEXT,target TEXT,source_doc TEXT,source_hash TEXT,
           citation INTEGER,passage TEXT)''',
        '''CREATE INDEX IF NOT EXISTS graph_edge_scope
           ON graph_edges(tenant_id,scope,generation_id,source)'''
    ):
        db.execute(sql)


def validate_records(nodes, edges):
    entities = {}
    for node in nodes:
        if node['type'] not in TYPES or not node['id'] or node['id'] in entities:
            raise ValueError('invalid_graph_node')
        if not node['aliases'] or not all(isinstance(a, str) and a.strip() for a in node['aliases']):
            raise ValueError('invalid_graph_alias')
        entities[node['id']] = node['type']
    for edge in edges:
        pair = (entities.get(edge['source']), entities.get(edge['target']))
        if pair != RELATIONS.get(edge['relation']) or not edge['passage'].strip():
            raise ValueError('invalid_graph_edge')


def refresh_graph(db):
    """Publish changed source-generation pairs inside the maintenance transaction.

    This small local implementation scans source metadata. A larger deployment
    can enqueue dirty source ids without changing the publication contract.
    """
    ensure_graph(db)
    documents = [dict(r) for r in db.execute('SELECT * FROM documents')]
    eligible = set()
    for document in documents:
        payload = json.loads(document['metadata']).get('graph')
        if payload is None:
            continue
        nodes, edges = payload['nodes'], payload['edges']
        validate_records(nodes, edges)
        generations = db.execute('''SELECT DISTINCT retrieval_generation_id FROM document_chunks
            WHERE document_id=? AND tenant_id=? AND source_hash=?
            AND chunker_version=? AND is_leaf=1''',
            (document['id'], document['tenant_id'], document['content_hash'],
             document['chunker_version'])).fetchall()
        for row in generations:
            gen = row[0]
            if gen is None:
                continue
            key = (document['id'], gen)
            eligible.add(key)
            signature = digest(canonical([document['content_hash'],
                document['metadata_hash'], document['source_scope'], document['chunker_version']]))
            old = db.execute('SELECT signature FROM graph_sources WHERE document_id=? AND generation_id=?', key).fetchone()
            if old and old[0] == signature:
                continue
            for table in ('graph_nodes', 'graph_edges', 'graph_sources'):
                db.execute(f'DELETE FROM {table} WHERE document_id=? AND generation_id=?', key)
            prefix = (document['tenant_id'], document['source_scope'], gen, document['id'])
            for node in nodes:
                db.execute('INSERT INTO graph_nodes VALUES(?,?,?,?,?,?,?,?)',
                    prefix + (node['id'], node['type'], canonical(node['aliases']), document['content_hash']))
            chunks = db.execute('''SELECT id,text FROM document_chunks WHERE document_id=?
                AND tenant_id=? AND retrieval_generation_id=? AND is_leaf=1
                AND source_hash=? AND chunker_version=? ORDER BY id''',
                (document['id'], document['tenant_id'], gen, document['content_hash'],
                 document['chunker_version'])).fetchall()
            for edge in edges:
                matches = [c for c in chunks if edge['passage'] in c['text']]
                if not matches:
                    raise ValueError('graph_passage_without_chunk')
                db.execute('INSERT INTO graph_edges VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    prefix + (edge['source'], edge['relation'], edge['target'],
                    document['external_id'], document['content_hash'], matches[0]['id'], edge['passage']))
            db.execute('INSERT INTO graph_sources VALUES(?,?,?)', key + (signature,))
    for row in db.execute('SELECT document_id,generation_id FROM graph_sources').fetchall():
        key = tuple(row)
        if key not in eligible:
            for table in ('graph_nodes', 'graph_edges', 'graph_sources'):
                db.execute(f'DELETE FROM {table} WHERE document_id=? AND generation_id=?', key)


def graph_snapshot(db, tenant, scope):
    valid_tenant(tenant)
    owns = not db.in_transaction
    if owns:
        db.execute('BEGIN')
    try:
        gen = active(db, tenant)
        result = dict(tenant_id=tenant, scope=scope,
                      generation_id=gen['id'] if gen else None, nodes=[], edges=[])
        if not gen or gen['status'] != 'ready':
            return result
        present = db.execute("SELECT 1 FROM sqlite_master WHERE name='graph_edges'").fetchone()
        if not present:
            return result
        args = (tenant, scope, gen['id'])
        nodes = db.execute('''SELECT n.* FROM graph_nodes n JOIN documents d
            ON d.id=n.document_id AND d.tenant_id=n.tenant_id
            WHERE n.tenant_id=? AND n.scope=? AND n.generation_id=?
            AND d.source_scope=n.scope AND d.status='ready' AND d.content_hash=n.source_hash
            ORDER BY n.node_id,n.document_id''', args).fetchall()
        result['nodes'] = [dict(n, aliases=json.loads(n['aliases'])) for n in nodes]
        edges = db.execute('''SELECT e.* FROM graph_edges e JOIN documents d
            ON d.id=e.document_id AND d.tenant_id=e.tenant_id
            JOIN document_chunks c ON c.id=e.citation AND c.document_id=d.id
            AND c.tenant_id=e.tenant_id AND c.retrieval_generation_id=e.generation_id
            WHERE e.tenant_id=? AND e.scope=? AND e.generation_id=?
            AND d.source_scope=e.scope AND d.status='ready' AND d.content_hash=e.source_hash
            AND c.source_hash=d.content_hash AND c.chunker_version=d.chunker_version
            AND c.is_leaf=1 ORDER BY e.source,e.target,e.citation''', args).fetchall()
        ids = {n['node_id'] for n in result['nodes']}
        result['edges'] = [dict(e) for e in edges
                           if e['source'] in ids and e['target'] in ids]
        return result
    finally:
        if owns:
            db.rollback()
