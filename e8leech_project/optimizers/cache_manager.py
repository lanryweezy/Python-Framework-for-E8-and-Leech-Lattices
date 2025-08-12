import time
from typing import Any, Dict, Optional
from threading import Lock

class CacheManager:
    """
    A simple in-memory cache manager with optional time-to-live (TTL) and size limits.
    """
    _cache: Dict[str, Dict[str, Any]] = {}
    _lock = Lock()
    _max_size: Optional[int] = None

    @staticmethod
    def configure(max_size: Optional[int] = None) -> None:
        with CacheManager._lock:
            CacheManager._max_size = max_size

    @staticmethod
    def set(key: str, value: Any, ttl: Optional[int] = None) -> None:
        with CacheManager._lock:
            if CacheManager._max_size is not None and len(CacheManager._cache) >= CacheManager._max_size:
                CacheManager._evict_oldest()

            expiration_time = time.time() + ttl if ttl is not None else None
            CacheManager._cache[key] = {"value": value, "expiration_time": expiration_time, "access_time": time.time()}

    @staticmethod
    def get(key: str) -> Optional[Any]:
        with CacheManager._lock:
            entry = CacheManager._cache.get(key)
            if entry is None:
                return None

            if entry["expiration_time"] is not None and time.time() > entry["expiration_time"]:
                del CacheManager._cache[key]
                return None

            entry["access_time"] = time.time()
            return entry["value"]

    @staticmethod
    def delete(key: str) -> None:
        with CacheManager._lock:
            if key in CacheManager._cache:
                del CacheManager._cache[key]

    @staticmethod
    def clear() -> None:
        with CacheManager._lock:
            CacheManager._cache.clear()

    @staticmethod
    def _evict_oldest() -> None:
        if not CacheManager._cache:
            return

        oldest_key = min(CacheManager._cache, key=lambda k: CacheManager._cache[k]["access_time"])
        del CacheManager._cache[oldest_key]

    @staticmethod
    def size() -> int:
        with CacheManager._lock:
            return len(CacheManager._cache)
