"""
Dermato-RAG - Utilities Modülü Testleri.

Config, logger ve helpers modüllerinin birim testleri.
"""

import os
from pathlib import Path

import pytest


class TestConfig:
    """Konfigürasyon yönetimi testleri."""

    def test_config_loads_successfully(self):
        """Konfigürasyon dosyaları başarıyla yüklenmeli."""
        from src.utils.config import Config

        # Singleton'ı sıfırla
        Config._instance = None
        Config._initialized = False

        config = Config()
        assert config is not None
        assert isinstance(config.all, dict)

    def test_config_singleton_pattern(self):
        """Config sınıfı singleton pattern ile çalışmalı."""
        from src.utils.config import Config

        Config._instance = None
        Config._initialized = False

        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_config_get_existing_key(self):
        """Mevcut bir anahtar değeri dönmeli."""
        from src.utils.config import Config

        Config._instance = None
        Config._initialized = False

        config = Config()
        project_name = config.get("project.name")
        assert project_name == "Dermato-RAG"

    def test_config_get_nonexistent_key(self):
        """Mevcut olmayan anahtar için default dönmeli."""
        from src.utils.config import Config

        Config._instance = None
        Config._initialized = False

        config = Config()
        result = config.get("nonexistent.key", "default_value")
        assert result == "default_value"

    def test_config_get_section(self):
        """Bölüm dictionary olarak dönmeli."""
        from src.utils.config import Config

        Config._instance = None
        Config._initialized = False

        config = Config()
        logging_section = config.get_section("logging")
        assert isinstance(logging_section, dict)
        assert "level" in logging_section

    def test_config_reload(self):
        """Config reload başarılı olmalı."""
        from src.utils.config import Config

        Config._instance = None
        Config._initialized = False

        config = Config()
        config.reload()  # Hata vermemeli
        assert config.get("project.name") == "Dermato-RAG"

    def test_deep_merge(self):
        """İki dict'in derin birleştirilmesi doğru çalışmalı."""
        from src.utils.config import _deep_merge

        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99, "e": 4}, "f": 5}
        result = _deep_merge(base, override)

        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["b"]["e"] == 4
        assert result["f"] == 5


class TestHelpers:
    """Yardımcı fonksiyonlar testleri."""

    def test_set_seed(self):
        """Seed ayarlama hata vermeden çalışmalı."""
        from src.utils.helpers import set_seed
        set_seed(42)  # Hata vermemeli

    def test_ensure_dir(self, tmp_path):
        """Dizin oluşturma doğru çalışmalı."""
        from src.utils.helpers import ensure_dir
        test_dir = tmp_path / "test_subdir" / "nested"
        result = ensure_dir(test_dir)
        assert result.exists()
        assert result.is_dir()

    def test_format_size(self):
        """Boyut formatlama doğru çalışmalı."""
        from src.utils.helpers import format_size

        assert "B" in format_size(500)
        assert "KB" in format_size(1024)
        assert "MB" in format_size(1024 * 1024)
        assert "GB" in format_size(1024 * 1024 * 1024)

    def test_get_project_root(self):
        """Proje kök dizini doğru dönmeli."""
        from src.utils.helpers import get_project_root
        root = get_project_root()
        assert root.exists()
        assert (root / "pyproject.toml").exists()

    def test_resolve_path_relative(self):
        """Göreceli yol doğru çözümlenmeli."""
        from src.utils.helpers import resolve_path
        path = resolve_path("configs/config.yaml")
        assert path.is_absolute()
        assert "configs" in str(path)

    def test_resolve_path_absolute(self):
        """Mutlak yol değiştirilmemeli."""
        import sys
        from src.utils.helpers import resolve_path

        if sys.platform == "win32":
            abs_path = Path("C:/some/absolute/path")
        else:
            abs_path = Path("/some/absolute/path")
        result = resolve_path(abs_path)
        assert result == abs_path

    def test_timer_context_manager(self):
        """Timer context manager hata vermeden çalışmalı."""
        import time
        from src.utils.helpers import timer

        with timer("Test işlemi"):
            time.sleep(0.01)  # Kısa bekleme


class TestLogger:
    """Loglama modülü testleri."""

    def test_get_logger_returns_logger(self):
        """get_logger bir Logger instance dönmeli."""
        from src.utils.logger import get_logger, _initialized_loggers

        # Cache'i temizle
        test_name = "test_logger_instance"
        _initialized_loggers.pop(test_name, None)

        logger = get_logger(test_name)
        assert logger is not None
        assert logger.name == test_name

    def test_logger_singleton_per_name(self):
        """Aynı isimle çağrılan logger aynı instance olmalı."""
        from src.utils.logger import get_logger, _initialized_loggers

        test_name = "test_singleton"
        _initialized_loggers.pop(test_name, None)

        logger1 = get_logger(test_name)
        logger2 = get_logger(test_name)
        assert logger1 is logger2

    def test_logger_has_handlers(self):
        """Logger'da en az bir handler olmalı."""
        from src.utils.logger import get_logger, _initialized_loggers

        test_name = "test_handlers"
        _initialized_loggers.pop(test_name, None)

        logger = get_logger(test_name)
        assert len(logger.handlers) >= 1

    def test_log_separator(self):
        """log_separator hata vermeden çalışmalı."""
        from src.utils.logger import get_logger, log_separator, _initialized_loggers

        test_name = "test_separator"
        _initialized_loggers.pop(test_name, None)

        logger = get_logger(test_name)
        log_separator(logger, "Test Başlık")  # Hata vermemeli

    def test_log_dict(self):
        """log_dict hata vermeden çalışmalı."""
        from src.utils.logger import get_logger, log_dict, _initialized_loggers

        test_name = "test_dict_log"
        _initialized_loggers.pop(test_name, None)

        logger = get_logger(test_name)
        log_dict(logger, {"key1": "value1", "key2": 42})  # Hata vermemeli
