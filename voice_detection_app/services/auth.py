"""API key authentication and rate limiting middleware."""
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10


@dataclass
class APIKeyConfig:
    key: str
    name: str
    tier: str = "standard"
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    enabled: bool = True


class APIKeyManager:
    """Manages API keys and their configurations."""

    def __init__(self):
        self.keys: dict[str, APIKeyConfig] = {}

    def add_key(self, key: str, name: str, tier: str = "standard", rate_limit: RateLimitConfig | None = None):
        self.keys[key] = APIKeyConfig(
            key=key,
            name=name,
            tier=tier,
            rate_limit=rate_limit or self._default_rate_limit(tier),
        )

    def validate_key(self, key: str) -> APIKeyConfig | None:
        config = self.keys.get(key)
        if config and config.enabled:
            return config
        return None

    def revoke_key(self, key: str) -> bool:
        if key in self.keys:
            self.keys[key].enabled = False
            return True
        return False

    def generate_key(self, name: str, tier: str = "standard") -> str:
        raw = f"{name}_{time.time()}_{os.urandom(8).hex()}"
        key = hashlib.sha256(raw.encode()).hexdigest()[:32]
        self.add_key(key, name, tier)
        return key

    @staticmethod
    def _default_rate_limit(tier: str) -> RateLimitConfig:
        limits = {
            "free": RateLimitConfig(10, 100, 2),
            "standard": RateLimitConfig(60, 1000, 10),
            "premium": RateLimitConfig(300, 5000, 50),
            "enterprise": RateLimitConfig(1000, 20000, 100),
        }
        return limits.get(tier, limits["standard"])


class RateLimiter:
    """Token bucket rate limiter per API key."""

    def __init__(self):
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._hourly: dict[str, list[float]] = defaultdict(list)

    def check_rate_limit(self, api_key: str, config: RateLimitConfig) -> bool:
        now = time.time()

        self._buckets[api_key] = [
            t for t in self._buckets[api_key] if now - t < 60
        ]
        if len(self._buckets[api_key]) >= config.requests_per_minute:
            return False

        self._hourly[api_key] = [
            t for t in self._hourly[api_key] if now - t < 3600
        ]
        if len(self._hourly[api_key]) >= config.requests_per_hour:
            return False

        recent = [t for t in self._buckets[api_key] if now - t < 1]
        if len(recent) >= config.burst_limit:
            return False

        self._buckets[api_key].append(now)
        self._hourly[api_key].append(now)
        return True

    def get_usage(self, api_key: str) -> dict:
        now = time.time()
        minute_count = len([t for t in self._buckets.get(api_key, []) if now - t < 60])
        hour_count = len([t for t in self._hourly.get(api_key, []) if now - t < 3600])
        return {
            "requests_this_minute": minute_count,
            "requests_this_hour": hour_count,
        }


class APIAuthMiddleware:
    """Authentication and rate limiting middleware."""

    def __init__(self):
        self.key_manager = APIKeyManager()
        self.rate_limiter = RateLimiter()
        self._setup_default_keys()

    def _setup_default_keys(self):
        self.key_manager.add_key(
            key="vd_dev_key_2024",
            name="Development",
            tier="standard",
        )

    async def authenticate(self, request: Request, api_key: str | None = Security(api_key_header)) -> APIKeyConfig:
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required. Pass it via X-API-Key header.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        config = self.key_manager.validate_key(api_key)
        if not config:
            raise HTTPException(status_code=401, detail="Invalid or revoked API key")

        if not self.rate_limiter.check_rate_limit(api_key, config.rate_limit):
            usage = self.rate_limiter.get_usage(api_key)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. {usage}",
                headers={
                    "X-RateLimit-Limit": str(config.rate_limit.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )

        return config


auth_middleware = APIAuthMiddleware()
