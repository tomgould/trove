"""
Health checking and circuit breaker for LM Studio integration.

CircuitBreaker prevents hammering a dead service — opens after N consecutive
failures, auto-probes for recovery after a cooldown period.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker pattern for LLM API calls.

    States:
        CLOSED: normal operation, requests pass through
        OPEN: service is down, requests are rejected immediately
        HALF_OPEN: after cooldown, allow one probe request
    """

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._total_rejected = 0

    def is_closed(self) -> bool:
        return self._consecutive_failures < self.failure_threshold

    def is_open(self) -> bool:
        return not self.is_closed()

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        if self.is_closed():
            return True

        # Check for half-open: cooldown expired -> allow one probe
        if self._opened_at and (time.time() - self._opened_at) >= self.cooldown_seconds:
            logger.info("CircuitBreaker: half-open, allowing probe request")
            return True

        self._total_rejected += 1
        return False

    def record_success(self):
        """Record a successful request — resets to CLOSED."""
        if self._consecutive_failures > 0:
            logger.info(
                f"CircuitBreaker: recovered after {self._consecutive_failures} failures "
                f"({self._total_rejected} requests were rejected)"
            )
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self):
        """Record a failed request — may trip to OPEN."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold and self._opened_at is None:
            self._opened_at = time.time()
            logger.error(
                f"CircuitBreaker: OPEN after {self._consecutive_failures} consecutive failures. "
                f"Will probe again in {self.cooldown_seconds}s."
            )

    @property
    def stats(self) -> dict:
        return {
            "state": "closed" if self.is_closed() else "open",
            "consecutive_failures": self._consecutive_failures,
            "total_rejected": self._total_rejected,
        }


def check_lm_studio_model(
    base_url: str = "http://localhost:1234/v1",
    timeout: float = 10.0,
) -> bool:
    """
    Verify LM Studio has a model loaded and can generate completions.

    Sends a tiny test prompt. Returns True only if we get a 200 with valid output.
    """
    try:
        import httpx
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Say OK"}],
                    "max_tokens": 5,
                    "temperature": 0,
                },
            )
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices and choices[0].get("message", {}).get("content"):
                    return True
            logger.warning(f"LM Studio health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"LM Studio health check failed: {e}")
        return False


def wait_for_lm_studio(
    base_url: str = "http://localhost:1234/v1",
    timeout: float = 60.0,
    poll_interval: float = 5.0,
) -> bool:
    """Poll LM Studio until a model is loaded or timeout."""
    start = time.time()
    while (time.time() - start) < timeout:
        if check_lm_studio_model(base_url):
            return True
        time.sleep(poll_interval)
    return False
