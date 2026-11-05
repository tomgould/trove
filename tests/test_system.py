"""
Imports, resource manager, model cache and health checks.
"""



# Test imports work
def test_imports():
    """Verify all core modules can be imported."""
    from core import client
    from core import ingest
    from core import llm
    from core import search
    from core import reference_config

    assert hasattr(client, 'Client')
    assert hasattr(client, 'load_client')
    assert hasattr(ingest, 'DocumentIngestor')
    assert hasattr(llm, 'LMStudioProvider')
    assert hasattr(llm, 'ClaudeProvider')
    assert hasattr(search, 'HybridSearcher')
    assert hasattr(search, 'EnhancedTokenizer')
    assert hasattr(reference_config, 'ReferenceConfig')


class TestResourceManager:
    """Tests for adaptive resource management."""

    def test_probe_cpu_returns_cores(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        info = rm.probe_cpu()
        assert "total_cores" in info
        assert "available_cores" in info
        assert info["available_cores"] <= info["total_cores"]
        assert info["available_cores"] > 0

    def test_probe_ram_returns_gb(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        info = rm.probe_ram()
        assert "total_gb" in info
        assert "available_gb" in info
        assert info["total_gb"] > 0

    def test_probe_gpu_returns_info_or_none(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        info = rm.probe_gpu()
        if info is not None:
            assert "total_mb" in info
            assert "free_mb" in info

    def test_recommend_parse_workers_scales_with_files(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        small = rm.recommend_parse_workers(100)
        large = rm.recommend_parse_workers(50000)
        assert large >= small
        assert small >= 2
        assert large <= rm.probe_cpu()["total_cores"]

    def test_recommend_embed_batch(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        batch = rm.recommend_embed_batch(384)
        assert 32 <= batch <= 1024

    def test_classify_client(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        assert rm.classify_client(100, 500) == "small"
        assert rm.classify_client(5000, 20000) == "medium"
        assert rm.classify_client(80000, 200000) == "large"

    def test_headroom_respected(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager(headroom=0.20)
        info = rm.probe_cpu()
        assert info["available_cores"] <= int(info["total_cores"] * 0.80) + 1

    def test_summary_returns_string(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        s = rm.summary()
        assert "CPU:" in s
        assert "RAM:" in s


class TestModelCache:
    """Tests for model caching."""

    def test_get_embedder_returns_model(self):
        from core.model_cache import ModelCache
        cache = ModelCache()
        model = cache.get_embedder("all-MiniLM-L6-v2", "cpu")
        assert model is not None
        assert hasattr(model, "encode")
        cache.clear()

    def test_get_embedder_returns_same_instance(self):
        from core.model_cache import ModelCache
        cache = ModelCache()
        model1 = cache.get_embedder("all-MiniLM-L6-v2", "cpu")
        model2 = cache.get_embedder("all-MiniLM-L6-v2", "cpu")
        assert model1 is model2
        cache.clear()

    def test_clear_removes_models(self):
        from core.model_cache import ModelCache
        cache = ModelCache()
        cache.get_embedder("all-MiniLM-L6-v2", "cpu")
        assert len(cache._embedders) > 0
        cache.clear()
        assert len(cache._embedders) == 0

    def test_singleton_access(self):
        import core.model_cache as mc
        mc._instance = None  # reset singleton
        c1 = mc.get_model_cache()
        c2 = mc.get_model_cache()
        assert c1 is c2
        mc._instance = None  # cleanup


class TestHealthCheck:
    """Tests for LM Studio health check and circuit breaker."""

    def test_circuit_breaker_starts_closed(self):
        from core.health import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.is_closed()

    def test_circuit_breaker_opens_after_threshold(self):
        from core.health import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_closed()
        cb.record_failure()
        assert cb.is_open()

    def test_circuit_breaker_resets_on_success(self):
        from core.health import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.is_closed()
        assert cb._consecutive_failures == 0

    def test_circuit_breaker_allows_check_returns_false_when_open(self):
        from core.health import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert not cb.allow_request()

    def test_circuit_breaker_stats(self):
        from core.health import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        stats = cb.stats
        assert stats["state"] == "open"
        assert stats["consecutive_failures"] == 2

    def test_check_model_loaded_handles_no_server(self):
        from core.health import check_lm_studio_model
        result = check_lm_studio_model(base_url="http://localhost:19999")
        assert result is False
