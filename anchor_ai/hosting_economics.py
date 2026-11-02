import math
from collections import defaultdict
from datetime import date

POLICIES = {
    'ticket_triage': {'warm_limit_ms': 1000, 'recovery_limit_ms': 10000,
                      'external_processing_assumed_approved': True},
    'kb_embedding': {'warm_limit_ms': 30000, 'recovery_limit_ms': 300000,
                     'external_processing_assumed_approved': True},
}
MISSING = ['dated prices and fully loaded costs', 'equivalent task quality',
           'processing boundary and tenant isolation audit',
           'measured warm/cold latency and recovery',
           'measured throughput, peak arrivals and failure capacity']


def require_production_routing(report):
    raise RuntimeError('Production routing disabled: reviewed real-evidence validator missing')


def compare(rows, config, utilization=0.2, operations_usd=180):
    if not math.isfinite(utilization) or not 0 < utilization <= 1:
        raise ValueError('utilization must be in (0, 1]')
    if not math.isfinite(operations_usd) or operations_usd < 0:
        raise ValueError('operations cost must be finite and nonnegative')
    groups = defaultdict(lambda: [0, 0])
    dates = set()
    for row in rows:
        dates.add(date.fromisoformat(row['date']))
        workload = row['workload']
        if workload not in POLICIES:
            raise ValueError('Unknown workload requires reviewed policy')
        for index, key in enumerate(('input_tokens', 'output_tokens')):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('token counts must be nonnegative integers')
            groups[workload][index] += value
    if not dates:
        raise ValueError('usage is empty')
    days = (max(dates) - min(dates)).days + 1
    result = {'production_routing_enabled': False,
              'basis': 'synthetic planning; no live performance verified',
              'observed_days': days, 'workloads': {}}
    local = config['self_hosted']
    hours = local['hours']
    headroom = local['headroom_fraction']
    if not 0 <= headroom < 1 or local['capacity_tokens_s'] <= 0 or hours <= 0:
        raise ValueError('invalid local capacity assumptions')
    for workload, counts in sorted(groups.items()):
        # Normalize partial calendar windows to the configured comparison month.
        inp, out = [n * (hours / 24) / days for n in counts]
        total = inp + out
        if total <= 0:
            raise ValueError('workload has no token demand')
        policy = POLICIES[workload]
        raw_capacity = local['capacity_tokens_s'] * hours * 3600
        usable = raw_capacity * utilization * (1 - headroom)
        replicas = max(1, math.ceil(total / usable))
        fixed = replicas * hours * local['hourly_usd'] + local['infrastructure_usd'] + operations_usd
        costs = {}
        matrix = []
        for option in ('hosted', 'managed', 'self_hosted'):
            assumptions = config[option]
            if option == 'self_hosted':
                cost = fixed + total / 1e6 * local['variable_per_million']
            else:
                cost = (inp * assumptions['input_per_million'] +
                        out * assumptions['output_per_million']) / 1e6 + assumptions['operations_usd']
            feasible = (assumptions['warm_p99_ms'] <= policy['warm_limit_ms'] and
                        assumptions['warm_p99_ms'] + assumptions['startup_ms'] <= policy['recovery_limit_ms'] and
                        (option == 'self_hosted' or policy['external_processing_assumed_approved']))
            costs[option] = cost
            matrix.append({'option': option, 'monthly_usd': round(cost, 2),
                           'effective_usd_per_million': cost / total * 1e6,
                           'planning_feasible': feasible, 'assumptions': dict(assumptions),
                           'latency_provenance': assumptions['latency_provenance'],
                           'privacy_constraints': assumptions['data_control'],
                           'operations_burden': assumptions['operations_burden'],
                           'missing_evidence': list(MISSING)})
        candidates = [r['option'] for r in matrix if r['planning_feasible']]
        choice = min(candidates, key=costs.get) if candidates else None
        sensitivity = []
        for allowance in sorted(set([0, operations_usd, 680])):
            one_fixed = hours * local['hourly_usd'] + local['infrastructure_usd'] + allowance
            for competitor in ('hosted', 'managed'):
                c = config[competitor]
                rate = (inp * c['input_per_million'] + out * c['output_per_million']) / total
                denominator = rate - local['variable_per_million']
                crossing = max(0, (one_fixed - c['operations_usd']) / denominator) if denominator > 0 else None
                sensitivity.append({'competitor': competitor, 'operations_usd': allowance,
                                    'fixed_replicas': 1, 'crossing_million_tokens': crossing,
                                    'one_replica_capacity_million': usable / 1e6,
                                    'fits_one_replica': crossing is not None and crossing <= usable / 1e6})
        result['workloads'][workload] = {
            'matrix': matrix, 'policy': dict(policy),
            'recommendation': {'option': choice, 'conditional': True,
                               'reason': 'Lowest estimated cost among planning-feasible options; real evidence required'},
            'capacity': {'replicas': replicas, 'utilization_assumption': utilization,
                         'headroom_fraction': headroom, 'usable_tokens': usable * replicas,
                         'spare_usable_tokens': usable * replicas - total,
                         'idle_capacity_fraction': 1 - total / (raw_capacity * replicas),
                         'infrastructure_usd': local['infrastructure_usd'],
                         'operations_usd': operations_usd, 'allocated_gpu_hours': hours * replicas},
            'break_even_sensitivity': sensitivity}
    return result
