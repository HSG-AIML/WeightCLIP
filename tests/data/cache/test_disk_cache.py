"""
Unit tests for DiskCache functionality.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import time
import threading

from sane.data.cache import DiskCache


class TestDiskCache:
    """Test suite for DiskCache functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a DiskCache instance for testing."""
        return DiskCache(cache_dir=temp_cache_dir)

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization and directory creation."""
        cache_dir = temp_cache_dir / "test_cache"
        cache = DiskCache(cache_dir=cache_dir, create_dirs=True)

        # Use resolve() to handle symlinks (e.g., macOS /var vs /private/var)
        assert cache.cache_dir.resolve() == cache_dir.resolve()
        assert cache_dir.exists()

    def test_basic_put_get(self, cache):
        """Test basic put and get operations."""
        # Create test data
        test_data = [
            (torch.randn(10, 5), torch.ones(10, 5, dtype=torch.bool), torch.randint(0, 100, (10, 3)))
        ]
        
        # Test put and get
        key = "test_checkpoint_1"
        cache.put(key, test_data)
        
        retrieved = cache.get(key)
        assert retrieved is not None
        assert len(retrieved) == 1
        
        # Compare tensors
        assert torch.equal(retrieved[0][0], test_data[0][0])
        assert torch.equal(retrieved[0][1], test_data[0][1])
        assert torch.equal(retrieved[0][2], test_data[0][2])

    def test_cache_miss(self, cache):
        """Test cache miss behavior."""
        result = cache.get("nonexistent_key")
        assert result is None
        
        result = cache.get("nonexistent_key", default="default_value")
        assert result == "default_value"

    def test_cache_contains(self, cache):
        """Test contains functionality."""
        key = "test_key"
        test_data = [torch.randn(5, 3)]
        
        assert not cache.contains(key)
        
        cache.put(key, test_data)
        assert cache.contains(key)

    def test_cache_removal(self, cache):
        """Test cache entry removal."""
        key = "test_key"
        test_data = [torch.randn(5, 3)]
        
        cache.put(key, test_data)
        assert cache.contains(key)
        
        cache.remove(key)
        assert not cache.contains(key)
        assert cache.get(key) is None

    def test_cache_statistics(self, cache):
        """Test cache statistics tracking."""
        # Perform some operations
        cache.put("key1", [torch.randn(3, 3)])
        cache.put("key2", [torch.randn(3, 3)])

        cache.get("key1")  # Hit
        cache.get("key1")  # Hit again
        cache.get("nonexistent")  # Miss

        stats = cache.get_stats()
        assert stats['writes'] >= 2
        assert stats['hits'] >= 2
        assert stats['misses'] >= 1
        assert 'hit_rate' in stats
        assert 'disk_cache_size_mb' in stats

    def test_cache_clear(self, cache):
        """Test cache clearing functionality."""
        # Add some data
        for i in range(3):
            cache.put(f"key_{i}", [torch.randn(2, 2)])

        # Verify data exists
        assert cache.contains("key_0")
        assert cache.get_stats()['disk_cache_files'] > 0

        # Clear cache
        cache.clear()

        # Verify cache is empty
        assert not cache.contains("key_0")
        assert cache.get_stats()['disk_cache_files'] == 0

    def test_thread_safety(self, cache):
        """Test thread-safe operations."""
        num_threads = 5
        ops_per_thread = 20
        results = {}
        
        def worker(thread_id):
            """Worker function for thread testing."""
            for i in range(ops_per_thread):
                key = f"thread_{thread_id}_key_{i}"
                data = [torch.randn(3, 3)]
                
                cache.put(key, data)
                retrieved = cache.get(key)
                
                assert retrieved is not None
                assert torch.equal(retrieved[0], data[0])
                results[key] = True
        
        # Start multiple threads
        threads = []
        for tid in range(num_threads):
            thread = threading.Thread(target=worker, args=(tid,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations completed successfully
        assert len(results) == num_threads * ops_per_thread

    def test_cache_key_generation(self, cache):
        """Test cache key generation for different input types."""
        # Test different key types
        keys = [
            "string_key",
            123,
            (1, 2, 3),
            ("tuple", "key", 456)
        ]
        
        for key in keys:
            cache_key = cache._get_cache_key(key)
            assert isinstance(cache_key, str)
            assert cache_key.startswith("cache_")
            assert cache_key.endswith(".pt")
        
        # Same key should generate same cache key
        assert cache._get_cache_key("test") == cache._get_cache_key("test")
        
        # Different keys should generate different cache keys
        assert cache._get_cache_key("test1") != cache._get_cache_key("test2")

    def test_corrupted_cache_handling(self, cache, temp_cache_dir):
        """Test handling of corrupted cache files."""
        key = "test_key"
        cache_path = cache._get_cache_path(key)
        
        # Create a corrupted file
        cache_path.write_text("corrupted data")
        
        # Should handle corruption gracefully
        result = cache.get(key)
        assert result is None
        
        # Corrupted file should be removed
        assert not cache_path.exists()

    def test_cache_info_string(self, cache):
        """Test human-readable cache info."""
        # Add some data
        cache.put("key1", [torch.randn(5, 5)])
        cache.get("key1")
        cache.get("nonexistent")

        info = cache.get_cache_info()
        assert isinstance(info, str)
        assert "DiskCache Stats" in info
        assert "hits/misses" in info.lower() or "hit rate" in info.lower()
        assert "disk cache" in info.lower()

    def test_empty_cache_dir(self, temp_cache_dir):
        """Test behavior with non-existent cache directory."""
        non_existent_dir = temp_cache_dir / "non_existent"
        
        # Should create directory when create_dirs=True
        cache = DiskCache(cache_dir=non_existent_dir, create_dirs=True)
        assert non_existent_dir.exists()
        
        # Should work normally
        cache.put("key", [torch.randn(2, 2)])
        assert cache.get("key") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])