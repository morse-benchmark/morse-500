"""Cumulative USD cap shared by every agent job that writes to one budget file.

Each upstream request first reserves its worst-case standard-rate cost (estimated
input + full output cap) under an exclusive lock; the request is refused if that
would exceed the limit. When the response ends, the reservation is settled to the
cost of the metered usage (same token rules as usage.py). Unsettled reservations
(e.g. a job killed mid-request) stay counted at their worst case.
"""

import fcntl
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import uuid
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimates.usage import PRICES, token_totals

# Conservative per-request input estimate: JSON text at 3 chars/token plus a
# generous flat allowance per embedded image (cached frames are <= 512 px).
CHARS_PER_TOKEN = 3
TOKENS_PER_IMAGE = 2000
# Base64 JPEG/PNG/GIF/WebP payloads (same signatures as trace redaction).
IMAGE_DATA = re.compile(r"(?:/9j/|iVBORw0KGgo|R0lGOD|UklGR)[A-Za-z0-9+/]{100,}={0,2}")
# Statuses whose metered usage may be incomplete.
PARTIAL_STATUSES = ("streaming", "timeout", "unknown", "received", "in_progress")
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def _thread_lock(path):
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(path.resolve()), threading.Lock())


class BudgetExceeded(Exception):
    """The shared USD limit leaves no room for another worst-case request."""


class Budget:
    def __init__(self, path, limit_usd, provider, model, prices_path=PRICES):
        if not math.isfinite(limit_usd) or limit_usd <= 0:
            raise ValueError("USD budget must be finite and positive")
        rates = json.loads(Path(prices_path).read_text())["models"].get(f"{provider}/{model}")
        if not rates:
            raise ValueError(f"No price for {provider}/{model} in {prices_path}; cannot enforce budget")
        for rate in (rates["input"], rates["output"]):
            if not math.isfinite(rate) or rate < 0:
                raise ValueError(f"Invalid price for {provider}/{model}")
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.thread_lock = _thread_lock(self.lock_path)
        self.limit = float(limit_usd)
        self.provider, self.model, self.rates = provider, model, rates
        self.owner = {
            "run_id": str(uuid.uuid4()), "job": os.environ.get("SLURM_JOB_ID"),
            "host": socket.gethostname(),
            "pid": os.getpid(), "provider": provider, "model": model,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def cost(self, input_tokens, output_tokens):
        return (input_tokens * self.rates["input"] + output_tokens * self.rates["output"]) / 1e6

    def usage_cost(self, usage):
        totals = token_totals({"provider": self.provider, "usage": usage})
        return None if totals is None else self.cost(*totals)

    def worst_case(self, body, max_output_tokens):
        text = json.dumps(body)
        images = len(IMAGE_DATA.findall(text))
        chars = len(IMAGE_DATA.sub("", text))
        return self.cost(chars // CHARS_PER_TOKEN + images * TOKENS_PER_IMAGE, max_output_tokens)

    @contextmanager
    def _locked(self):
        with self.thread_lock:
            with self.lock_path.open("a") as stream:
                fcntl.lockf(stream, fcntl.LOCK_EX)  # POSIX lock: honored across NFS clients.
                try:
                    yield
                finally:
                    fcntl.lockf(stream, fcntl.LOCK_UN)

    def _read(self):
        reserved, settled = {}, {}
        if self.path.exists():
            with self.path.open() as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        if not line.endswith("\n"):
                            continue  # Torn final line from a killed writer.
                        raise ValueError(f"Invalid budget ledger JSON: {self.path}:{number}")
                    (reserved if event["type"] == "reserve" else settled)[event["id"]] = event["usd"]
        open_usd = sum(usd for key, usd in reserved.items() if key not in settled)
        return sum(settled.values()), open_usd

    def _append(self, event):
        event["time"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def spent(self):
        """(settled USD, still-reserved USD) across all jobs sharing the file."""
        with self._locked():
            return self._read()

    def require(self, usd):
        settled, open_usd = self.spent()
        if settled + open_usd + usd > self.limit:
            raise BudgetExceeded(
                f"USD budget reached: ${settled + open_usd:.2f} committed of "
                f"${self.limit:.2f}; next worst-case request needs ${usd:.2f}"
            )

    def reserve(self, usd, example_id=None):
        """Atomically reserve `usd`; returns an ID, or None if it would exceed the limit."""
        with self._locked():
            settled, open_usd = self._read()
            if settled + open_usd + usd > self.limit:
                return None
            key = str(uuid.uuid4())
            self._append(dict(self.owner, type="reserve", id=key, usd=usd, example_id=example_id))
            return key

    def settle(self, key, reserved_usd, usage, status, upstream_accepted):
        """Replace a reservation by metered cost; keep the worst case if usage may be partial."""
        actual = self.usage_cost(usage) if usage is not None else None
        if not upstream_accepted:
            usd = actual or 0.0  # Refused/failed before the provider accepted work.
        elif actual is None or status in PARTIAL_STATUSES or status.startswith("gateway_error_"):
            usd = max(actual or 0.0, reserved_usd)
        else:
            usd = actual
        with self._locked():
            self._append(dict(self.owner, type="settle", id=key, usd=usd,
                              metered_usd=actual, status=status))
        return usd
