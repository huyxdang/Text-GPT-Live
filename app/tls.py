"""Opt-in system trust store for TLS.

Some networks (e.g. corporate firewalls doing SSL inspection) re-sign HTTPS
traffic with a CA that the OS trusts but Python's bundled certifi does not —
and Python 3.13+ strict validation additionally rejects forged leaf certs
that omit the Authority Key Identifier. Setting SYSTEM_TLS=1 makes
Python verify certificates through the operating system (as curl and
browsers do) via the `truststore` package. Verification stays on.
"""

from __future__ import annotations

import os


def maybe_use_system_certs() -> None:
    if os.environ.get("SYSTEM_TLS") != "1":
        return
    import truststore

    truststore.inject_into_ssl()
