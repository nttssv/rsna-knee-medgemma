"""Validate read/stop access to an already stopped Pod, without starting compute."""
import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request


class PreflightError(RuntimeError):
    """Safe error message: never contains credentials or authenticated URLs."""


def request(query, token):
    # RunPod's documented GraphQL authentication. Never log this URL.
    url = 'https://api.runpod.io/graphql?' + urllib.parse.urlencode({'api_key': token})
    req = urllib.request.Request(url, data=json.dumps({'query': query}).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise PreflightError(f'Provider HTTP {exc.code}') from None
    except (OSError, ValueError):
        raise PreflightError('Provider request failed; details suppressed to protect credentials') from None
    if not isinstance(payload, dict) or payload.get('errors'):
        raise PreflightError('Provider rejected the operation')
    return payload.get('data', {})


def check(pod_id, token, transport=request):
    if not pod_id or not pod_id.isalnum():
        raise PreflightError('Expected a nonempty alphanumeric Pod ID')
    if not token.strip():
        raise PreflightError('Missing RSNA_RUNPOD_CONTROL_TOKEN')
    target = json.dumps(pod_id)
    data = transport('query { pod(input: {podId: ' + target + '}) { id desiredStatus } }', token)
    pod = data.get('pod')
    if not isinstance(pod, dict) or pod.get('id') != pod_id or pod.get('desiredStatus') != 'EXITED':
        raise PreflightError('Target identity/stopped state unverified; no stop operation sent')
    data = transport('mutation { podStop(input: {podId: ' + target + '}) { id desiredStatus } }', token)
    pod = data.get('podStop')
    if not isinstance(pod, dict) or pod.get('id') != pod_id or pod.get('desiredStatus') != 'EXITED':
        raise PreflightError('Provider did not confirm the target remains stopped')
    return {'pod_id': pod_id, 'read_verified': True, 'stop_accepted_on_stopped_pod': True,
            'desired_status': 'EXITED', 'started_compute': False,
            'future_running_pod_stop_guaranteed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod-id', required=True)
    parser.add_argument('--confirm-stop-already-stopped', action='store_true')
    args = parser.parse_args()
    if not args.confirm_stop_already_stopped:
        parser.error('Explicit confirmation of the bounded stopped-pod operation is required')
    try:
        result = check(args.pod_id, os.environ.get('RSNA_RUNPOD_CONTROL_TOKEN', ''))
    except PreflightError as exc:
        parser.exit(1, str(exc) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
