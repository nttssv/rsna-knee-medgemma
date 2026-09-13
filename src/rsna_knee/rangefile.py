"""Bounded HTTP range reader; never downloads the full archive."""

import io
import requests


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.position = 0
        self.bytes_read = 0
        with self.session.get(
            url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60
        ) as r:
            if r.status_code != 206:
                raise RuntimeError(f"Range download unsupported: HTTP {r.status_code}")
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            if r.content != b"P":
                raise RuntimeError("Expected ZIP archive signature")
        print(
            "Archive supports selective downloads; total archive GB:",
            round(self.size / 1000000000.0, 2),
            flush=True,
        )

    def seek(self, offset, whence=0):
        self.position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.size + offset
        )
        if not 0 <= self.position <= self.size:
            raise ValueError("Invalid ZIP seek")
        return self.position

    def tell(self):
        return self.position

    def read(self, length=-1):
        if length < 0:
            length = self.size - self.position
        length = min(length, self.size - self.position)
        if not length:
            return b""
        if length > 350000000 or self.bytes_read + length > 400000000:
            raise RuntimeError("ZIP metadata exceeded the 400 MB metadata budget")
        start = self.position
        with self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{start + length - 1}"},
            stream=True,
            timeout=120,
        ) as r:
            if r.status_code != 206:
                raise RuntimeError(f"Range read failed: HTTP {r.status_code}")
            value = r.content
        if len(value) != length:
            raise RuntimeError("Incomplete ZIP metadata range")
        self.position += length
        self.bytes_read += length
        return value
