"""
RAG Chain Caching Service

This module provides caching functionality for RAG chains to improve performance
by avoiding rebuilding chains for each query. Chains are cached per user and
can be pre-built during login or lazily loaded on first use.
"""

import asyncio
import time
from typing import Dict, Tuple, Optional
from threading import Lock
import logging
from datetime import datetime, timedelta

from app.services.rag_service import create_rag_chain
from app.utils.embeddings import get_embeddings

logger = logging.getLogger(__name__)

class ChainCacheEntry:
    """Represents a cached RAG chain with metadata"""
    
    def __init__(self, rag_chain, knowledge_base, username: str):
        self.rag_chain = rag_chain
        self.knowledge_base = knowledge_base
        self.username = username
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.access_count = 0

    def update_access(self):
        """Update last accessed time and increment access count"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def is_expired(self, max_age_minutes: int = 60) -> bool:
        """Check if the cache entry has expired"""
        return datetime.utcnow() - self.created_at > timedelta(minutes=max_age_minutes)

class RAGChainCache:
    """Thread-safe cache for RAG chains with automatic cleanup"""
    
    def __init__(self, max_cache_size: int = 100, max_age_minutes: int = 60):
        self._cache: Dict[str, ChainCacheEntry] = {}
        self._lock = Lock()
        self._max_cache_size = max_cache_size
        self._max_age_minutes = max_age_minutes
        self._embeddings = None
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_task())

    def _get_cache_key(self, username: str) -> str:
        """Generate cache key for user"""
        return f"chain_{username}"

    def _get_embeddings(self):
        """Get embeddings provider (cached)"""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    async def get_or_create_chain(self, username: str) -> Tuple[Optional[object], Optional[object]]:
        """
        Get cached chain or create new one if not exists/expired
        
        Returns:
            Tuple of (rag_chain, knowledge_base) or (None, None) if error
        """
        cache_key = self._get_cache_key(username)
        
        with self._lock:
            # Check if we have a valid cached entry
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                if not entry.is_expired(self._max_age_minutes):
                    entry.update_access()
                    logger.info(f"Cache hit for user {username} (accessed {entry.access_count} times)")
                    return entry.rag_chain, entry.knowledge_base
                else:
                    # Remove expired entry
                    logger.info(f"Removing expired cache entry for user {username}")
                    del self._cache[cache_key]

        # Cache miss or expired - create new chain
        logger.info(f"Cache miss for user {username}, creating new chain")
        return await self._create_and_cache_chain(username)

    async def _create_and_cache_chain(self, username: str) -> Tuple[Optional[object], Optional[object]]:
        """Create new chain and add to cache"""
        try:
            # Create chain in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = self._get_embeddings()
            
            rag_chain, knowledge_base = await loop.run_in_executor(
                None, 
                create_rag_chain, 
                embeddings, 
                username
            )
            
            if rag_chain is None or knowledge_base is None:
                logger.error(f"Failed to create RAG chain for user {username}")
                return None, None

            cache_key = self._get_cache_key(username)
            entry = ChainCacheEntry(rag_chain, knowledge_base, username)
            
            with self._lock:
                # Check cache size and remove oldest entries if needed
                if len(self._cache) >= self._max_cache_size:
                    self._remove_oldest_entries()
                
                self._cache[cache_key] = entry
                logger.info(f"Cached new RAG chain for user {username}")

            return rag_chain, knowledge_base
            
        except Exception as e:
            logger.error(f"Error creating and caching chain for user {username}: {str(e)}")
            return None, None

    async def get_or_create_streaming_chain(self, username: str) -> Tuple[Optional[object], Optional[object]]:
        """
        Get cached streaming chain or create new one if not exists/expired
        
        Returns:
            Tuple of (streaming_chain, knowledge_base) or (None, None) if error
        """
        cache_key = f"streaming_chain_{username}"
        
        with self._lock:
            # Check if we have a valid cached entry
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                if not entry.is_expired(self._max_age_minutes):
                    entry.update_access()
                    logger.info(f"Streaming cache hit for user {username} (accessed {entry.access_count} times)")
                    return entry.rag_chain, entry.knowledge_base
                else:
                    # Remove expired entry
                    logger.info(f"Removing expired streaming cache entry for user {username}")
                    del self._cache[cache_key]

        # Cache miss or expired - create new streaming chain
        logger.info(f"Streaming cache miss for user {username}, creating new chain")
        return await self._create_and_cache_streaming_chain(username)

    async def _create_and_cache_streaming_chain(self, username: str) -> Tuple[Optional[object], Optional[object]]:
        """Create new streaming chain and add to cache"""
        try:
            # Import here to avoid circular imports
            from app.services.rag_service import create_rag_chain_streaming
            
            # Create streaming chain in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = self._get_embeddings()
            
            streaming_chain, knowledge_base = await loop.run_in_executor(
                None, 
                create_rag_chain_streaming, 
                embeddings, 
                username
            )
            
            if streaming_chain is None or knowledge_base is None:
                logger.error(f"Failed to create streaming RAG chain for user {username}")
                return None, None

            cache_key = f"streaming_chain_{username}"
            entry = ChainCacheEntry(streaming_chain, knowledge_base, username)
            
            with self._lock:
                # Check cache size and remove oldest entries if needed
                if len(self._cache) >= self._max_cache_size:
                    self._remove_oldest_entries()
                
                self._cache[cache_key] = entry
                logger.info(f"Cached new streaming RAG chain for user {username}")

            return streaming_chain, knowledge_base
            
        except Exception as e:
            logger.error(f"Error creating and caching streaming chain for user {username}: {str(e)}")
            return None, None

    def _remove_oldest_entries(self, count: int = 1):
        """Remove oldest cache entries to make space"""
        if not self._cache:
            return
            
        # Sort by last accessed time and remove oldest
        sorted_entries = sorted(
            self._cache.items(), 
            key=lambda x: x[1].last_accessed
        )
        
        for i in range(min(count, len(sorted_entries))):
            key_to_remove = sorted_entries[i][0]
            removed_entry = self._cache.pop(key_to_remove)
            logger.info(f"Removed cache entry for user {removed_entry.username} due to size limit")

    async def pre_warm_cache(self, username: str) -> bool:
        """
        Pre-warm cache for a user (useful during login)
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Pre-warming cache for user {username}")
        rag_chain, knowledge_base = await self._create_and_cache_chain(username)
        return rag_chain is not None and knowledge_base is not None

    def invalidate_user_cache(self, username: str):
        """Remove user's cache entry"""
        cache_key = self._get_cache_key(username)
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.info(f"Invalidated cache for user {username}")

    def clear_cache(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            logger.info("Cleared all cache entries")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            stats = {
                "total_entries": len(self._cache),
                "max_size": self._max_cache_size,
                "max_age_minutes": self._max_age_minutes,
                "entries": []
            }
            
            for key, entry in self._cache.items():
                stats["entries"].append({
                    "username": entry.username,
                    "created_at": entry.created_at.isoformat(),
                    "last_accessed": entry.last_accessed.isoformat(),
                    "access_count": entry.access_count,
                    "age_minutes": (datetime.utcnow() - entry.created_at).total_seconds() / 60
                })
            
            return stats

    async def _cleanup_task(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                with self._lock:
                    expired_keys = [
                        key for key, entry in self._cache.items()
                        if entry.is_expired(self._max_age_minutes)
                    ]
                    
                    for key in expired_keys:
                        removed_entry = self._cache.pop(key)
                        logger.info(f"Cleaned up expired cache entry for user {removed_entry.username}")
                        
                    if expired_keys:
                        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                        
            except Exception as e:
                logger.error(f"Error in cache cleanup task: {str(e)}")

# Global cache instance
_chain_cache = None

def get_chain_cache() -> RAGChainCache:
    """Get the global chain cache instance"""
    global _chain_cache
    if _chain_cache is None:
        _chain_cache = RAGChainCache(max_cache_size=100, max_age_minutes=60)
    return _chain_cache
