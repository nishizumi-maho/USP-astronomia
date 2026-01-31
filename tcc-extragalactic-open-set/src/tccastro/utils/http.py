from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, rate_limit_rps: float) -> None:
        self.rate_limit_rps = rate_limit_rps
        self._lock = threading.Lock()
        self._last_time = 0.0

    def wait(self) -> None:
        if self.rate_limit_rps <= 0:
            return
        with self._lock:
            now = time.monotonic()
            min_interval = 1.0 / self.rate_limit_rps
            elapsed = now - self._last_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_time = time.monotonic()


@dataclass
class HttpClient:
    timeout_s: int = 20
    retries: int = 3
    backoff_s: float = 1.0
    rate_limit_rps: float = 2.0

    def __post_init__(self) -> None:
        self._rate_limiter = RateLimiter(self.rate_limit_rps)

    def get(self, url: str) -> bytes:
        for attempt in range(self.retries + 1):
            try:
                self._rate_limiter.wait()
                response = requests.get(url, timeout=self.timeout_s)
                if response.status_code != 200:
                    raise requests.HTTPError(
                        f"Status {response.status_code} for {url}"
                    )
                return response.content
            except requests.RequestException as exc:
                if attempt >= self.retries:
                    logger.error("Failed to fetch %s after %s attempts", url, attempt + 1)
                    raise
                sleep_time = self.backoff_s * (2**attempt)
                logger.warning("Retrying %s in %.2fs: %s", url, sleep_time, exc)
                time.sleep(sleep_time)
        raise RuntimeError("Unexpected retry loop exit")
