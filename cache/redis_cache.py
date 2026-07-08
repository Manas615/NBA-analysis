"""
Redis Cache — TTL-based caching for expensive computations.

Caches trade simulations, player projections, championship simulations,
and other costly operations with configurable TTLs.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import wraps
from typing import Any, Callable

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# TTL constants (seconds)
CACHE_TTL_TRADE = 3600        # 1 hour
CACHE_TTL_PLAYER = 21600      # 6 hours
CACHE_TTL_CHAMPIONSHIP = 1800  # 30 minutes
CACHE_TTL_ROSTER = 3600       # 1 hour
CACHE_TTL_DEFAULT = 1800      # 30 minutes


class RedisCache:
    """Redis cache client with typed get/set operations."""

    def __init__(self, url: str | None = None):
        self._url = url or REDIS_URL
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis | None:
        """Lazy-initialize Redis connection."""
        if self._client is None:
            try:
                self._client = redis.from_url(
                    self._url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                self._client.ping()
            except (redis.ConnectionError, redis.TimeoutError, OSError):
                self._client = None
        return self._client

    @property
    def available(self) -> bool:
        """Check if Redis is available."""
        return self.client is not None

    def get(self, key: str) -> Any | None:
        """Get a cached value by key."""
        if not self.available:
            return None
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
        except (redis.RedisError, json.JSONDecodeError):
            pass
        return None

    def set(self, key: str, value: Any, ttl: int = CACHE_TTL_DEFAULT) -> bool:
        """Set a cached value with TTL."""
        if not self.available:
            return False
        try:
            self.client.setex(key, ttl, json.dumps(value, default=str))
            return True
        except redis.RedisError:
            return False

    def delete(self, key: str) -> bool:
        """Delete a cached key."""
        if not self.available:
            return False
        try:
            self.client.delete(key)
            return True
        except redis.RedisError:
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        if not self.available:
            return 0
        try:
            keys = list(self.client.scan_iter(match=pattern, count=100))
            if keys:
                return self.client.delete(*keys)
        except redis.RedisError:
            pass
        return 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self.available:
            return {"available": False}
        try:
            info = self.client.info("stats")
            return {
                "available": True,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0)
                    / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                ),
            }
        except redis.RedisError:
            return {"available": False}


def _make_cache_key(prefix: str, args: tuple, kwargs: dict) -> str:
    """Create a deterministic cache key from function arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_data.encode()).hexdigest()
    return f"nba:{prefix}:{key_hash}"


def cached(prefix: str, ttl: int = CACHE_TTL_DEFAULT) -> Callable:
    """
    Decorator to cache function results in Redis.

    Usage:
        @cached("trade_sim", ttl=3600)
        def simulate_trade(team_a, player_a, team_b, player_b):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            key = _make_cache_key(prefix, args, kwargs)

            # Try cache first
            cached_result = cache.get(key)
            if cached_result is not None:
                cached_result["_cache_hit"] = True
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            if isinstance(result, dict):
                cache.set(key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# Global cache instance
_cache: RedisCache | None = None


def get_cache() -> RedisCache:
    """Get the global Redis cache instance."""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache
