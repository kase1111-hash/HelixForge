"""Phase 6: Production hardening tests.

Tests for:
  - Custom exception hierarchy (utils/errors.py)
  - Configuration validation (utils/config.py)
  - Structured API error responses (api/server.py)
  - Health check endpoint
"""

import os
import tempfile

import pytest
import yaml


# ------------------------------------------------------------------ #
#  Exception hierarchy                                                 #
# ------------------------------------------------------------------ #

class TestExceptionHierarchy:
    """All domain exceptions inherit from HelixForgeError."""

    def test_base_exception(self):
        from utils.errors import HelixForgeError
        exc = HelixForgeError("test message", detail="extra info")
        assert exc.message == "test message"
        assert exc.detail == "extra info"
        assert str(exc) == "test message"
        assert isinstance(exc, Exception)

    def test_ingestion_error_inherits(self):
        from utils.errors import HelixForgeError, IngestionError
        exc = IngestionError("bad file")
        assert isinstance(exc, HelixForgeError)

    def test_configuration_error_inherits(self):
        from utils.errors import ConfigurationError, HelixForgeError
        exc = ConfigurationError("bad config")
        assert isinstance(exc, HelixForgeError)

    def test_alignment_error_inherits(self):
        from utils.errors import AlignmentError, HelixForgeError
        exc = AlignmentError("no match")
        assert isinstance(exc, HelixForgeError)

    def test_fusion_error_inherits(self):
        from utils.errors import FusionError, HelixForgeError
        exc = FusionError("merge failed")
        assert isinstance(exc, HelixForgeError)

    def test_insight_error_inherits(self):
        from utils.errors import HelixForgeError, InsightError
        exc = InsightError("stats failed")
        assert isinstance(exc, HelixForgeError)

    def test_interpretation_error_inherits(self):
        from utils.errors import HelixForgeError, InterpretationError
        exc = InterpretationError("interpret failed")
        assert isinstance(exc, HelixForgeError)

    def test_ingestor_uses_shared_error(self):
        """DataIngestorAgent imports IngestionError from utils.errors."""
        from agents.data_ingestor_agent import IngestionError
        from utils.errors import HelixForgeError
        assert issubclass(IngestionError, HelixForgeError)


# ------------------------------------------------------------------ #
#  Configuration validation                                            #
# ------------------------------------------------------------------ #

class TestConfigValidation:
    """Config validation catches errors at startup with clear messages."""

    def setup_method(self):
        from utils.config import reset_config
        reset_config()

    def test_valid_config_loads(self):
        from utils.config import load_config, reset_config
        reset_config()
        config = load_config()
        assert isinstance(config, dict)

    def test_defaults_provided(self):
        """All optional keys have sensible defaults."""
        from utils.config import AppConfig
        cfg = AppConfig()
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4o"
        assert cfg.api.port == 8000
        assert cfg.logging.level == "INFO"
        assert cfg.processing.max_file_size_mb == 500

    def test_invalid_log_level_rejected(self):
        from utils.config import validate_config
        from utils.errors import ConfigurationError
        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            validate_config({"logging": {"level": "VERBOSE"}})

    def test_invalid_log_format_rejected(self):
        from utils.config import validate_config
        from utils.errors import ConfigurationError
        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            validate_config({"logging": {"format": "xml"}})

    def test_negative_port_rejected(self):
        from utils.config import validate_config
        from utils.errors import ConfigurationError
        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            validate_config({"api": {"port": -1}})

    def test_invalid_strategy_rejected(self):
        from utils.config import validate_config
        from utils.errors import ConfigurationError
        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            validate_config({"fusion": {"default_join_strategy": "magic"}})

    def test_invalid_imputation_rejected(self):
        from utils.config import validate_config
        from utils.errors import ConfigurationError
        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            validate_config({"fusion": {"imputation_method": "neural"}})

    def test_temperature_bounds(self):
        from utils.config import validate_config
        from utils.errors import ConfigurationError
        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            validate_config({"llm": {"temperature": 5.0}})

    def test_malformed_yaml_raises(self):
        from utils.config import load_config, reset_config
        from utils.errors import ConfigurationError
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{ bad yaml [[[")
            f.flush()
            try:
                with pytest.raises(ConfigurationError, match="Failed to parse"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)
                reset_config()

    def test_non_dict_yaml_raises(self):
        from utils.config import load_config, reset_config
        from utils.errors import ConfigurationError
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- list\n- not\n- a mapping\n")
            f.flush()
            try:
                with pytest.raises(ConfigurationError, match="YAML mapping"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)
                reset_config()

    def test_env_override_string(self, monkeypatch):
        from utils.config import reset_config, load_config
        reset_config()
        monkeypatch.setenv("HELIXFORGE_LLM__MODEL", "gpt-3.5-turbo")
        config = load_config()
        assert config["llm"]["model"] == "gpt-3.5-turbo"
        reset_config()

    def test_env_override_int(self, monkeypatch):
        from utils.config import reset_config, load_config
        reset_config()
        monkeypatch.setenv("HELIXFORGE_API__PORT", "9000")
        config = load_config()
        assert config["api"]["port"] == 9000
        reset_config()

    def test_env_override_bool(self, monkeypatch):
        from utils.config import reset_config, load_config
        reset_config()
        monkeypatch.setenv("HELIXFORGE_LOGGING__LEVEL", "DEBUG")
        config = load_config()
        assert config["logging"]["level"] == "DEBUG"
        reset_config()

    def test_missing_config_file_uses_defaults(self):
        from utils.config import load_config, reset_config
        reset_config()
        config = load_config("/nonexistent/config.yaml")
        assert isinstance(config, dict)
        reset_config()

    def test_reset_clears_cache(self):
        from utils.config import reset_config, load_config
        reset_config()
        c1 = load_config()
        reset_config()
        c2 = load_config()
        # Should not be the same object since cache was reset
        assert c1 is not c2


# ------------------------------------------------------------------ #
#  Health check                                                        #
# ------------------------------------------------------------------ #

class TestHealthCheck:
    """The /health endpoint reports dependency status."""

    @pytest.fixture
    def test_client(self):
        from api.server import app
        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")

    @pytest.mark.asyncio
    async def test_health_returns_200(self, test_client):
        async with test_client as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")
        assert "checks" in body

    @pytest.mark.asyncio
    async def test_health_reports_openai_key(self, test_client, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        async with test_client as client:
            resp = await client.get("/health")
        body = resp.json()
        assert body["checks"]["openai_key"] == "missing"

    @pytest.mark.asyncio
    async def test_root_endpoint(self, test_client):
        async with test_client as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "HelixForge API"
        assert body["version"] == "1.0.0"
