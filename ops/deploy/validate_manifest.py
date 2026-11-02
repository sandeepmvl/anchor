import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FILES = ('compose.yml', 'ops/deploy/anchor-production.yml')
SECRET_KEYS = {'SECRET_KEY_BASE', 'ANCHOR_AI_SECRET'}


def selected_route(*, environ=None, require_secret=False):
    env = os.environ if environ is None else environ
    configured = env.get('ANCHOR_DEPLOYMENT_CONFIG')
    if not configured:
        raise ValueError('missing routing configuration')
    path = Path(configured)
    if not path.is_absolute():
        path = ROOT / path
    try:
        policy = yaml.safe_load(path.read_text())['deployment']
        name = env.get('ANCHOR_DEPLOY_ROUTE', policy['default'])
        route = policy['routes'][name]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise ValueError('invalid routing configuration') from error
    if route.get('enabled') is not True:
        raise ValueError('route is not released')
    # Offline evidence cannot release a live provider, even after a flag edit.
    if name != 'fake' or route.get('provider') != 'offline':
        raise ValueError('live provider release evidence required')
    if require_secret and not env.get('ANCHOR_AI_SECRET', '').strip():
        raise ValueError('missing service authentication secret')
    return {'name': name, 'provider': route['provider']}


def inference_client():
    from anchor_ai.llm_client import LlmClient
    route = selected_route(require_secret=True)
    return LlmClient(provider=route['provider'])


def launch_policy(manifest):
    launch = manifest.get('x-anchor-launch', {})
    require(launch.get('entrypoint') == 'bin/deploy', 'launch entrypoint missing')
    require(launch.get('production_evidence') == 'required', 'production evidence must be required')
    require(set(launch.get('independent_versions', [])) == {'prompt', 'model', 'index'},
            'independent release versions missing')
    require(set(launch.get('required_controls', [])) == {
        'tenancy', 'auth', 'quotas', 'redaction', 'guarded_tools', 'tracing',
        'rollback', 'backup_restore', 'kill_switch'}, 'launch controls missing')
    return launch


def release_gate():
    raise ValueError('release refused: container and live-provider evidence missing')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def render(environ=None):
    env = {} if environ is None else environ
    result = {}
    for name in FILES:
        result = merge(result, yaml.safe_load((ROOT / name).read_text()))
    # Secret placeholders remain placeholders: no secrets appear in output.
    for service in result['services'].values():
        for key, value in service.get('environment', {}).items():
            if key in SECRET_KEYS:
                continue
            if isinstance(value, str):
                match = re.fullmatch(r'\$\{([A-Z_]+):-([^}]+)\}', value)
                if match:
                    service['environment'][key] = env.get(match[1], match[2])
    return result


def validate(manifest):
    launch_policy(manifest)
    services = manifest.get('services', {})
    require(set(services) == {'rails', 'ai-runtime', 'database', 'worker'},
            'required services missing')
    require(manifest['networks']['private'].get('internal') is True,
            'private network must be internal')
    for name, service in services.items():
        require('private' in service.get('networks', []), 'private communication missing')
        require(service.get('healthcheck', {}).get('test'), 'readiness check missing')
        require(service['healthcheck'].get('timeout'), 'health timeout missing')
        require(service.get('profiles') == ['offline-production'], 'offline profile missing')
        require(not service.get('privileged'), 'privileged service forbidden')
        require(service.get('network_mode') != 'host', 'host network forbidden')
        require(not service.get('container_name'), 'fixed container name prevents scaling')
        for mount in service.get('volumes', []):
            require('docker.sock' not in str(mount), 'runtime socket forbidden')
        if name != 'rails':
            require(not service.get('ports'), 'private service publishes a host port')
        for key, value in service.get('environment', {}).items():
            if key in SECRET_KEYS:
                require(value == '${' + key + ':?supply ' + key + '}',
                        'embedded credential or optional secret')
            elif re.search(r'(PASSWORD|TOKEN|API_KEY|SECRET)', key):
                raise ValueError('unrecognized credential setting')
    rails = services['rails']
    ai = services['ai-runtime']
    worker = services['worker']
    require(rails['ports'] == ['127.0.0.1:3000:3000'], 'unexpected Rails exposure')
    require(rails['environment']['ANCHOR_AI_URL'] == 'http://ai-runtime:8000',
            'AI service address missing')
    require(rails['environment']['DATABASE_URL'] ==
            'sqlite3:/rails/storage/production.sqlite3', 'shared database missing')
    require(worker['environment'] == rails['environment'], 'worker configuration differs')
    require(worker['command'] == ['./bin/jobs'], 'durable worker missing')
    require('solid_queue_jobs' in str(rails['healthcheck']['test']),
            'Rails readiness must check queue schema')
    require('last_heartbeat_at' in str(worker['healthcheck']['test']),
            'worker readiness must check heartbeat')
    require(worker['depends_on']['rails']['condition'] == 'service_healthy',
            'worker starts before Rails preparation')
    require(rails['depends_on']['database']['condition'] == 'service_healthy',
            'database readiness dependency missing')
    require(rails['depends_on']['ai-runtime']['condition'] == 'service_healthy',
            'AI readiness dependency missing')
    require(ai.get('read_only') is True and ai.get('cap_drop') == ['ALL'],
            'AI hardening missing')
    require(ai.get('security_opt') == ['no-new-privileges:true'], 'AI escalation protection missing')
    require(ai['environment'].get('ANCHOR_LLM_PROVIDER') == 'offline',
            'fake provider default missing')
    route = selected_route(environ=ai['environment'])
    require(rails['build']['target'] == 'rails-runtime', 'Rails image target missing')
    require(ai['build']['target'] == 'ai-runtime', 'AI image target missing')
    require(rails['deploy']['replicas'] == 1 and ai['deploy']['replicas'] == 2,
            'independent replica settings missing')
    return route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', action='store_true')
    args = parser.parse_args()
    import os
    route = validate(render(os.environ))
    if args.release:
        release_gate()
    print(json.dumps({'manifest': 'valid', 'route': route,
                      'services': ['rails', 'ai-runtime', 'database', 'worker'],
                      'container_deployment': 'not verified',
                      'release': 'not authorized'}))


if __name__ == '__main__':
    main()
