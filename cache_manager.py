
import time
from typing import Any, Dict, Optional
from threading import Lock
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class CacheManager:
    """
    A simple in-memory cache manager with optional time-to-live (TTL) and size limits.
    """
    _cache: Dict[str, Dict[str, Any]] = {}
    _lock = Lock()
    _max_size: Optional[int] = None

    @staticmethod
    def configure(max_size: Optional[int] = None) -> None:
        """
        Configures the cache manager.

        Args:
            max_size: The maximum number of items the cache can hold. None for no limit.
        """
        with CacheManager._lock:
            CacheManager._max_size = max_size
            logger.info(f"CacheManager configured with max_size={max_size}")

    @staticmethod
    def set(key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Sets a value in the cache.

        Args:
            key: The key for the cache entry.
            value: The value to store.
            ttl: Time-to-live in seconds. After this time, the entry will be considered expired.
                 None for no expiration.
        """
        with CacheManager._lock:
            if CacheManager._max_size is not None and len(CacheManager._cache) >= CacheManager._max_size:
                CacheManager._evict_oldest() # Evict oldest if cache is full
            
            expiration_time = time.time() + ttl if ttl is not None else None
            CacheManager._cache[key] = {"value": value, "expiration_time": expiration_time, "access_time": time.time()}
            logger.debug(f"CacheManager: Set key \'{key}\' with TTL {ttl}")

    @staticmethod
    def get(key: str) -> Optional[Any]:
        """
        Retrieves a value from the cache.

        Args:
            key: The key for the cache entry.

        Returns:
            The cached value, or None if the key is not found or expired.
        """
        with CacheManager._lock:
            entry = CacheManager._cache.get(key)
            if entry is None:
                logger.debug(f"CacheManager: Key \'{key}\' not found.")
                return None

            if entry["expiration_time"] is not None and time.time() > entry["expiration_time"]:
                del CacheManager._cache[key]
                logger.debug(f"CacheManager: Key \'{key}\' expired and removed.")
                return None
            
            entry["access_time"] = time.time() # Update access time for LRU-like behavior
            logger.debug(f"CacheManager: Retrieved key \'{key}\'")
            return entry["value"]

    @staticmethod
    def delete(key: str) -> None:
        """
        Deletes a key-value pair from the cache.

        Args:
            key: The key to delete.
        """
        with CacheManager._lock:
            if key in CacheManager._cache:
                del CacheManager._cache[key]
                logger.debug(f"CacheManager: Deleted key \'{key}\'")

    @staticmethod
    def clear() -> None:
        """
        Clears all entries from the cache.
        """
        with CacheManager._lock:
            CacheManager._cache.clear()
            logger.info("CacheManager: Cache cleared.")

    @staticmethod
    def _evict_oldest() -> None:
        """
        Evicts the oldest entry from the cache (least recently accessed).
        """
        if not CacheManager._cache:
            return
        
        oldest_key = None
        oldest_access_time = float("inf")
        
        for key, entry in CacheManager._cache.items():
            if entry["access_time"] < oldest_access_time:
                oldest_access_time = entry["access_time"]
                oldest_key = key
        
        if oldest_key:
            del CacheManager._cache[oldest_key]
            logger.debug(f"CacheManager: Evicted oldest key \'{oldest_key}\'")

    @staticmethod
    def size() -> int:
        """
        Returns the current number of items in the cache.
        """
        with CacheManager._lock:
            return len(CacheManager._cache)


# Example usage and testing
if __name__ == "__main__":
    print("--- CacheManager Tests ---")

    # Test basic set and get
    CacheManager.set("my_data", {"value": 123})
    print(f"Retrieved my_data: {CacheManager.get('my_data')}")
    assert CacheManager.get("my_data") == {"value": 123}

    # Test TTL
    CacheManager.set("temp_data", "hello", ttl=1)
    print(f"Retrieved temp_data (before expiration): {CacheManager.get('temp_data')}")
    time.sleep(1.1) # Wait for expiration
    print(f"Retrieved temp_data (after expiration): {CacheManager.get('temp_data')}")
    assert CacheManager.get("temp_data") is None

    # Test max size
    CacheManager.configure(max_size=2)
    CacheManager.set("key1", "value1")
    CacheManager.set("key2", "value2")
    print(f"Cache size: {CacheManager.size()}")
    assert CacheManager.size() == 2

    CacheManager.set("key3", "value3") # This should evict key1
    print(f"Cache size after adding key3: {CacheManager.size()}")
    print(f"Retrieved key1 (should be None): {CacheManager.get('key1')}")
    assert CacheManager.get("key1") is None
    assert CacheManager.get("key2") == "value2"
    assert CacheManager.get("key3") == "value3"

    # Test delete
    CacheManager.delete("key2")
    print(f"Retrieved key2 after delete (should be None): {CacheManager.get('key2')}")
    assert CacheManager.get("key2") is None

    # Test clear
    CacheManager.set("simple_key", "simple_value")
    print(f"Retrieved simple_key: {CacheManager.get('simple_key')}")
    CacheManager.clear()
    print(f"Retrieved simple_key after clear (should be None): {CacheManager.get('simple_key')}")
    assert CacheManager.get("simple_key") is None
    
    print("All CacheManager tests passed!")


