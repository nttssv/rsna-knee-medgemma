import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error

spec = importlib.util.spec_from_file_location('stopped_preflight', Path(__file__).parents[1]/'scripts/runpod_stopped_preflight.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PreflightTests(unittest.TestCase):
    def test_only_queries_then_stops_exact_already_stopped_target(self):
        calls = []
        def fake(query, token):
            calls.append(query)
            key = 'pod' if query.startswith('query') else 'podStop'
            return {key: {'id': 'abc123', 'desiredStatus': 'EXITED'}}
        result = m.check('abc123', 'fixture', fake)
        self.assertTrue(result['stop_accepted_on_stopped_pod'])
        self.assertFalse(result['future_running_pod_stop_guaranteed'])
        self.assertEqual(len(calls), 2)
        self.assertIn('podStop', calls[1])
        self.assertNotIn('podResume', ''.join(calls))
        self.assertTrue(all('"abc123"' in c for c in calls))

    def test_running_or_wrong_identity_never_sends_mutation(self):
        for pod in ({'id': 'abc123', 'desiredStatus': 'RUNNING'},
                    {'id': 'other', 'desiredStatus': 'EXITED'}, None):
            calls = []
            def fake(query, token):
                calls.append(query)
                return {'pod': pod}
            with self.assertRaises(m.PreflightError): m.check('abc123', 'fixture', fake)
            self.assertEqual(len(calls), 1)

    def test_missing_token_and_invalid_id_fail_before_network(self):
        for pod, token in [('abc123', ''), ('bad"id', 'fixture')]:
            with self.assertRaises(m.PreflightError):
                m.check(pod, token, lambda *_: self.fail('Network called'))

    def test_stop_response_must_confirm_exact_target_and_state(self):
        for response in ({'id': 'other', 'desiredStatus': 'EXITED'},
                         {'id': 'abc123', 'desiredStatus': 'RUNNING'}, None):
            def fake(query, token):
                return {'pod': {'id': 'abc123', 'desiredStatus': 'EXITED'}} if query.startswith('query') else {'podStop': response}
            with self.assertRaises(m.PreflightError): m.check('abc123', 'fixture', fake)

    def test_http_failure_does_not_expose_secret_url(self):
        with patch.object(m.urllib.request, 'urlopen', side_effect=urllib.error.HTTPError('https://invalid/?api_key=SECRET', 403, 'SECRET', {}, None)):
            with self.assertRaises(m.PreflightError) as error: m.request('query{}', 'SECRET')
        self.assertEqual(str(error.exception), 'Provider HTTP 403')

    def test_network_failure_does_not_expose_secret(self):
        with patch.object(m.urllib.request, 'urlopen', side_effect=OSError('SECRET')):
            with self.assertRaises(m.PreflightError) as error: m.request('query{}', 'SECRET')
        self.assertNotIn('SECRET', str(error.exception))

    def test_explicit_application_user_agent(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"data": {"pod": {"id": "abc123"}}}'
        with patch.object(m.urllib.request, 'urlopen', return_value=Response()) as send:
            m.request('query{}', 'fixture')
        self.assertEqual(send.call_args.args[0].get_header('User-agent'), 'rsna-preflight/1.0')
