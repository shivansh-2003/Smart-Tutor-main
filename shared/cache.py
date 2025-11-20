"""
Local cache manager for Smart Tutor
Handles caching of RAG results, LLM responses, and embeddings
"""

import os
import json
import time
import hashlib
import pickle
from typing import Any, Optional, Dict
from pathlib import Path

from core.config import get_config


class CacheManager:
    """Disk-based cache manager for local storage"""
    
    def __init__(self, cache_namespace: str = "default"):
        self.config = get_config()
        self.namespace = cache_namespace
        self.enabled = self.config.cache.enabled
        
        # Set cache directory based on namespace
        cache_dirs = {
            "rag": self.config.paths.cache_rag_dir,
            "llm": self.config.paths.cache_llm_dir,
            "embeddings": self.config.paths.cache_embeddings_dir,
            "default": self.config.paths.cache_dir
        }
        
        self.cache_dir = cache_dirs.get(cache_namespace, self.config.paths.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # TTL mapping
        self.ttl_map = {
            "rag": self.config.cache.rag_results_ttl,
            "llm": self.config.cache.llm_responses_ttl,
            "embeddings": self.config.cache.embeddings_ttl,
            "default": self.config.cache.default_ttl
        }
        
        self.default_ttl = self.ttl_map.get(cache_namespace, self.config.cache.default_ttl)
    
    def _generate_key(self, key: str) -> str:
        """Generate cache key hash"""
        if self.config.cache.enable_query_normalization:
            key = key.lower().strip()
        
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key"""
        key_hash = self._generate_key(key)
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path for cache key"""
        key_hash = self._generate_key(key)
        return self.cache_dir / f"{key_hash}.meta"
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Set cache value
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        if not self.enabled:
            return False
        
        try:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            # Save value
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Save metadata
            cache_metadata = {
                "key": key,
                "created_at": time.time(),
                "ttl": ttl or self.default_ttl,
                "namespace": self.namespace,
                "metadata": metadata or {}
            }
            
            with open(meta_path, 'w') as f:
                json.dump(cache_metadata, f)
            
            return True
        
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cache value
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.enabled:
            return None
        
        try:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            if not cache_path.exists() or not meta_path.exists():
                return None
            
            # Check TTL
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            created_at = metadata.get("created_at", 0)
            ttl = metadata.get("ttl", self.default_ttl)
            
            if time.time() - created_at > ttl:
                # Expired, delete
                self.delete(key)
                return None
            
            # Load value
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            
            return value
        
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        try:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            if cache_path.exists():
                cache_path.unlink()
            
            if meta_path.exists():
                meta_path.unlink()
            
            return True
        
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return self.get(key) is not None
    
    def clear_namespace(self) -> int:
        """Clear all cache entries in current namespace"""
        try:
            count = 0
            for file_path in self.cache_dir.glob("*.cache"):
                meta_path = file_path.with_suffix(".meta")
                
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata.get("namespace") == self.namespace:
                        file_path.unlink()
                        meta_path.unlink()
                        count += 1
            
            return count
        
        except Exception as e:
            print(f"Cache clear error: {e}")
            return 0
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries"""
        try:
            count = 0
            current_time = time.time()
            
            for meta_path in self.cache_dir.glob("*.meta"):
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    created_at = metadata.get("created_at", 0)
                    ttl = metadata.get("ttl", self.default_ttl)
                    
                    if current_time - created_at > ttl:
                        cache_path = meta_path.with_suffix(".cache")
                        
                        if cache_path.exists():
                            cache_path.unlink()
                        
                        meta_path.unlink()
                        count += 1
                
                except Exception:
                    continue
            
            return count
        
        except Exception as e:
            print(f"Cache cleanup error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            total_size = 0
            total_count = 0
            expired_count = 0
            namespace_count = 0
            current_time = time.time()
            
            for file_path in self.cache_dir.glob("*.cache"):
                total_size += file_path.stat().st_size
                total_count += 1
                
                meta_path = file_path.with_suffix(".meta")
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check namespace
                    if metadata.get("namespace") == self.namespace:
                        namespace_count += 1
                    
                    # Check expiration
                    created_at = metadata.get("created_at", 0)
                    ttl = metadata.get("ttl", self.default_ttl)
                    
                    if current_time - created_at > ttl:
                        expired_count += 1
            
            return {
                "namespace": self.namespace,
                "total_entries": total_count,
                "namespace_entries": namespace_count,
                "expired_entries": expired_count,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_dir": str(self.cache_dir)
            }
        
        except Exception as e:
            print(f"Cache stats error: {e}")
            return {}


def get_cache(namespace: str = "default") -> CacheManager:
    """Get cache manager instance"""
    return CacheManager(namespace)


# Decorator for caching function results
def cached(namespace: str = "default", ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator for caching function results
    
    Args:
        namespace: Cache namespace
        ttl: Time to live
        key_prefix: Prefix for cache key
    """
    def decorator(func):
        cache = get_cache(namespace)
        
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(filter(None, key_parts))
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator