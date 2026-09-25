"""Local synthetic intake checks: no real credentials, no RunPod API."""
import importlib.util
from pathlib import Path
import queue
import re
import stat
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location("intake", Path(__file__).resolve().parents[1] / "scripts/receive_runpod_key.py")
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)
DUMMY = "rpa_" + "SYNTHETIC_NOT_A_REAL_KEY" * 2


class IntakeTests(unittest.TestCase):
    def test_private_no_overwrite_and_no_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"secret.key"
            intake.save_secret(p, DUMMY)
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError): intake.save_secret(p, DUMMY)
            link = Path(d)/"link.key"; link.symlink_to(p)
            with self.assertRaises(FileExistsError): intake.save_secret(link, DUMMY)

    def test_invalid_format_not_saved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"key"
            for value in ["", "not-a-key", "rpa_"+"x"*300, DUMMY+"\nextra"]:
                with self.assertRaises(ValueError): intake.save_secret(p,value)
            self.assertFalse(p.exists())

    def test_same_origin_nonce_single_save_and_no_echo(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"key"; q=queue.Queue()
            thread=threading.Thread(target=intake.serve,args=(p,),kwargs={"seconds":5,"ready":q.put},daemon=True)
            thread.start(); url=q.get(timeout=2)
            self.assertTrue(url.startswith("http://127.0.0.1:"))
            with urlopen(url,timeout=2) as r:
                page=r.read().decode(); self.assertEqual(r.headers['Cache-Control'],'no-store')
            nonce=re.search('name="nonce" value="([^"]+)"',page).group(1)
            def post(origin, token):
                return urlopen(Request(url+'save',data=urlencode({'nonce':token,'key':DUMMY}).encode(),headers={'Origin':origin,'Content-Type':'application/x-www-form-urlencoded'}),timeout=2)
            for origin,token in [('https://untrusted.invalid',nonce),(url.rstrip('/'),'wrong')]:
                with self.assertRaises(HTTPError) as caught: post(origin,token)
                caught.exception.close()
                self.assertFalse(p.exists())
            with post(url.rstrip('/'),nonce) as r: self.assertNotIn(DUMMY,r.read().decode())
            thread.join(timeout=2); self.assertFalse(thread.is_alive())
            self.assertEqual(p.read_text().strip(),DUMMY)

    def test_expiry_without_key(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"key"
            self.assertFalse(intake.serve(p,seconds=0,ready=lambda _:None))
            self.assertFalse(p.exists())


if __name__ == '__main__': unittest.main()
