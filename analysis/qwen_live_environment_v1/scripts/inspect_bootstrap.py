"""Inspect pinned OCI startup artifacts without running or extracting an image.

Only small layers are fetched. Skipped layers explicitly prevent final-filesystem
or no-model-startup claims. Registry bearer tokens remain in worker memory.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import urllib.parse
import urllib.request

from observe_provider import load_frozen, private_write, sha

MANIFEST = 'sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851'
CONFIG = 'sha256:02b731f844d2fbc4dd0de87253081363864327c7931b1ae2b5daa60abc600c51'
REGISTRY = 'https://registry-1.docker.io/v2/runpod/pytorch/'
LAYER_LIMIT = 128 * 1024
TARGETS = {'opt/nvidia/nvidia_entrypoint.sh', 'start.sh'}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(url, limit, token=None):
    headers = {'User-Agent': 'rsna-qwen-bootstrap-audit/1.0',
               'Accept': 'application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        response = opener.open(urllib.request.Request(url, headers=headers), timeout=10)
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 307, 308):
            raise ValueError('Public registry request failed') from None
        redirected = urllib.parse.urlsplit(error.headers.get('Location', ''))
        if (redirected.scheme != 'https' or redirected.username or redirected.password
                or redirected.port not in (None, 443) or redirected.fragment
                or not (redirected.hostname == 'production.cloudflare.docker.com'
                        or (redirected.hostname or '').endswith('.cloudflarestorage.com')
                        or (redirected.hostname or '').endswith('.docker.com'))):
            raise ValueError('Untrusted registry redirect') from None
        # NEVER forward the registry bearer to a blob CDN.
        response = opener.open(urllib.request.Request(redirected.geturl(),
            headers={'User-Agent': headers['User-Agent']}), timeout=10)
    with response:
        if response.status != 200:
            raise ValueError('Public registry request failed')
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Registry object exceeds byte bound')
    return raw


def verify_blob(raw, digest, size=None):
    if 'sha256:' + sha(raw) != digest or (size is not None and len(raw) != size):
        raise ValueError('Pinned OCI object hash/size mismatch')


def inspect_layer(raw, digest, size):
    verify_blob(raw, digest, size)
    found, hazards = [], []
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r|gz') as archive:
        for index, member in enumerate(archive):
            if index > 10000:
                raise ValueError('Layer entry bound exceeded')
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive member')
            name = str(path)
            # Keep all whiteouts and links visible; never reconstruct a merged
            # filesystem or follow a tar link on the operator host.
            if path.name.startswith('.wh.'):
                hazards.append({'path': name, 'kind': 'whiteout'})
            target = name in TARGETS or name.startswith('opt/nvidia/entrypoint.d/')
            if member.issym() or member.islnk():
                hazards.append({'path': name, 'kind': 'link', 'target': member.linkname})
            if target and not member.isdir():
                if not member.isfile() or member.size > 65536:
                    raise ValueError('Startup artifact is not a bounded regular file')
                content = archive.extractfile(member).read(65537)
                found.append({'path': name, 'sha256': sha(content),
                              'mode': member.mode, 'size': len(content),
                              'content': content.decode('utf-8')})
    return {'digest': digest, 'size': size, 'startup_artifacts': found, 'hazards': hazards}


def inspect():
    auth = json.loads(fetch('https://auth.docker.io/token?service=registry.docker.io&scope=repository:runpod/pytorch:pull', 65536))
    token = auth['token']
    manifest_raw = fetch(REGISTRY + 'manifests/' + MANIFEST, 1024 * 1024, token)
    verify_blob(manifest_raw, MANIFEST)
    manifest = json.loads(manifest_raw)
    if manifest['config']['digest'] != CONFIG:
        raise ValueError('Unexpected image config')
    config_raw = fetch(REGISTRY + 'blobs/' + CONFIG, 1024 * 1024, token)
    verify_blob(config_raw, CONFIG, manifest['config']['size'])
    config = json.loads(config_raw)
    if config.get('architecture') != 'amd64' or config.get('os') != 'linux':
        raise ValueError('Unexpected image platform')
    startup = config['config']
    if startup.get('Entrypoint') != ['/opt/nvidia/nvidia_entrypoint.sh'] or startup.get('Cmd') != ['/start.sh']:
        raise ValueError('Pinned startup changed')
    layers, skipped = [], []
    for index, layer in enumerate(manifest['layers']):
        if layer['size'] > LAYER_LIMIT:
            skipped.append({'index': index, 'digest': layer['digest'], 'size': layer['size']})
            continue
        raw = fetch(REGISTRY + 'blobs/' + layer['digest'], LAYER_LIMIT, token)
        result = inspect_layer(raw, layer['digest'], layer['size'])
        layers.append({'index': index, **result})
    return {'schema': 'qwen_bootstrap_inspection_v1', 'observed_at': datetime.now(timezone.utc).isoformat(),
        'manifest_digest': MANIFEST, 'config_digest': CONFIG,
        'entrypoint': startup['Entrypoint'], 'image_default_cmd': startup['Cmd'],
        'requested_cmd': ['/bin/sleep', 'infinity'],
        'documented_effective_argv': startup['Entrypoint'] + ['/bin/sleep', 'infinity'],
        'inspected_layers': layers, 'skipped_layers': skipped,
        'all_layers_inspected': not skipped, 'live_command_mapping_verified': False,
        'no_automatic_model_activity_verified': False, 'ready_for_live_use': False,
        'blockers': ['Final merged filesystem/dependencies not verified; large layers omitted.',
                     'RunPod effective command mapping has not been observed on the pinned allocation.',
                     'No image/container executed; no empirical absence-of-model-activity evidence.']}


def main():
    if sys.argv[1:] == ['--worker']:
        try:
            result = {'ok': True, 'inspection': inspect()}
        except Exception:
            result = {'ok': False, 'error': 'bootstrap_inspection_failed'}
        print(json.dumps(result))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    network = load_frozen('network_supervisor')
    result = network.supervise([sys.executable, str(Path(__file__).resolve()), '--worker'], {},
        timeout=60, deadline=datetime.now(timezone.utc) + timedelta(seconds=60))
    if not result.get('ok'):
        raise ValueError('Bootstrap inspection failed; nothing verified')
    private_write(args.output, result['inspection'])
    print(json.dumps({k: result['inspection'][k] for k in ['manifest_digest', 'all_layers_inspected', 'ready_for_live_use']}))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
