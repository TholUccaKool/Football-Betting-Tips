"""Monkey-patch soccerdata to use plain ``requests.Session`` instead of ``tls_requests.Client``.

Why this exists
---------------
soccerdata 1.9.0 switched its HTTP layer to ``tls_requests``, a TLS-fingerprint-
spoofing library.  football-data.co.uk's reverse proxy rejects the TLS fingerprint
that ``tls_requests`` presents, returning HTTP 503 ("The page is temporarily
unavailable"), while plain ``requests`` and ``curl`` work fine.

The patch replaces ``BaseRequestsReader._init_session`` so it returns a standard
``requests.Session`` with a browser-like User-Agent.  All of soccerdata's
parsing, caching, and team-name normalisation logic is left untouched.

Verified against: soccerdata==1.9.0
Revisit / remove if soccerdata ships a fix upstream.
"""

import requests as _requests

from soccerdata._common import BaseRequestsReader

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _patched_init_session(self, headers=None):
    session = _requests.Session()
    session.headers.update({"User-Agent": _BROWSER_UA})
    if headers:
        session.headers.update(headers)
    return session


def patch_soccerdata_session():
    """Replace BaseRequestsReader._init_session with a plain-requests version."""
    BaseRequestsReader._init_session = _patched_init_session
