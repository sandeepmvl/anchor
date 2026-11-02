from collections import defaultdict, deque
from .graph_index import graph_snapshot


def retrieve_snapshot(snapshot, question, terms, max_depth=3, max_expansions=100):
    """Pure traversal; the caller must supply an authorized snapshot."""
    if len(terms) != 2 or not 0 <= max_depth <= 8 or not 1 <= max_expansions <= 10000:
        raise ValueError('invalid_graph_limits')
    aliases = defaultdict(set)
    for node in snapshot['nodes']:
        for alias in node['aliases']:
            aliases[alias.casefold()].add(node['node_id'])
    linked = {}
    for term in terms:
        matches = aliases[term.casefold()]
        linked[term] = next(iter(matches)) if len(matches) == 1 else None
    report = dict(question=question, tenant_id=snapshot['tenant_id'], scope=snapshot['scope'],
                  generation_id=snapshot['generation_id'], linked_ids=linked,
                  link_errors=[term for term in terms if linked[term] is None],
                  status='unlinked', paths=[], expansions=0, truncated=False)
    if report['link_errors']:
        return report
    adjacency = defaultdict(list)
    for edge in snapshot['edges']:
        adjacency[edge['source']].append(edge)
    start, goal = [linked[term] for term in terms]
    queue, visited = deque([(start, [])]), {start}
    report['status'] = 'unsupported'
    while queue:
        current, path = queue.popleft()
        if current == goal:
            report['status'] = 'supported'
            report['paths'] = [path]
            return report
        if len(path) >= max_depth:
            report['truncated'] = True
            continue
        if report['expansions'] >= max_expansions:
            report['truncated'] = True
            break
        report['expansions'] += 1
        for edge in adjacency[current]:
            if edge['target'] not in visited:
                visited.add(edge['target'])
                queue.append((edge['target'], path + [edge]))
    return report


class GraphRetriever:
    def __init__(self, db, tenant_id, scope, max_depth=3, max_expansions=100):
        self.db, self.tenant_id, self.scope = db, tenant_id, scope
        self.max_depth, self.max_expansions = max_depth, max_expansions

    def retrieve(self, question, terms):
        snapshot = graph_snapshot(self.db, self.tenant_id, self.scope)
        return retrieve_snapshot(snapshot, question, terms,
                                 self.max_depth, self.max_expansions)


def path_answer(report):
    """Deterministic evidence rendering, not a model-quality evaluation."""
    if report['status'] != 'supported':
        return dict(reply='I could not find a supporting policy in your documents.',
                    citations=[], abstained=True)
    path = report['paths'][0]
    ids = [report['linked_ids'][next(iter(report['linked_ids']))]]
    ids.extend(edge['target'] for edge in path)
    citations = list(dict.fromkeys(edge['citation'] for edge in path))
    if not citations:
        return dict(reply='I could not find a supporting policy in your documents.',
                    citations=[], abstained=True)
    return dict(reply='Supported relationship: ' + ' → '.join(ids),
                citations=citations, abstained=False)
