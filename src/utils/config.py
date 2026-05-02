"""
Dermato-RAG Konfigürasyon Yönetimi.

YAML tabanlı konfigürasyon dosyalarını yükler ve yönetir.
Ortam değişkenleri ile override desteği sağlar.

Kullanım:
    from src.utils.config import load_config, get_config

    # Tüm konfigürasyonları yükle
    config = load_config()

    # Belirli bir konfigürasyonu al
    model_cfg = get_config("model")
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# .env dosyasını yükle
load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(filepath: Path) -> Dict[str, Any]:
    """
    YAML dosyasını yükler ve dict olarak döndürür.

    Args:
        filepath: YAML dosyasının yolu.

    Returns:
        Dict olarak YAML içeriği.

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
        yaml.YAMLError: YAML parse hatası.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
            return data if data is not None else {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML parse hatası ({filepath}): {e}") from e


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    İki dictionary'yi derin birleştirme yapar.
    Override'daki değerler base'dekileri geçersiz kılar.

    Args:
        base: Temel dictionary.
        override: Geçersiz kılacak dictionary.

    Returns:
        Birleştirilmiş dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ortam değişkenlerini konfigürasyona uygular.
    Format: DERMATO_RAG__{SECTION}__{KEY} = value

    Örnek:
        DERMATO_RAG__LOGGING__LEVEL=DEBUG  →  config["logging"]["level"] = "DEBUG"
        DERMATO_RAG__DEVICE__USE_GPU=false →  config["device"]["use_gpu"] = False

    Args:
        config: Mevcut konfigürasyon dictionary.

    Returns:
        Ortam değişkenleri uygulanmış konfigürasyon.
    """
    prefix = "DERMATO_RAG__"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        # Anahtar parçalarını çıkar
        parts = env_key[len(prefix):].lower().split("__")

        # Değeri uygun tipe dönüştür
        converted_value: Any
        if env_value.lower() in ("true", "yes", "1"):
            converted_value = True
        elif env_value.lower() in ("false", "no", "0"):
            converted_value = False
        elif env_value.isdigit():
            converted_value = int(env_value)
        else:
            try:
                converted_value = float(env_value)
            except ValueError:
                converted_value = env_value

        # Nested dictionary'ye yerleştir
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = converted_value

    return config


class Config:
    """
    Merkezi konfigürasyon yönetici sınıfı.

    Singleton pattern ile tek bir global instance sağlar.
    Tüm YAML konfigürasyonlarını birleştirir ve ortam
    değişkenleri ile override edilmesine izin verir.
    """

    _instance: Optional["Config"] = None
    _initialized: bool = False

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._merged: Dict[str, Any] = {}
        self._load_all()
        self._initialized = True

    def _load_all(self) -> None:
        """Tüm konfigürasyon dosyalarını yükler ve birleştirir."""
        config_files = {
            "main": "config.yaml",
            "model": "model_config.yaml",
            "data": "data_config.yaml",
        }

        for name, filename in config_files.items():
            filepath = CONFIGS_DIR / filename
            if filepath.exists():
                self._configs[name] = _load_yaml(filepath)
            else:
                self._configs[name] = {}

        # Tüm config'leri birleştir
        self._merged = {}
        for config_data in self._configs.values():
            self._merged = _deep_merge(self._merged, config_data)

        # Ortam değişkenleri ile override
        self._merged = _apply_env_overrides(self._merged)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Nokta notasyonu ile konfigürasyon değerini alır.

        Args:
            key: Nokta ile ayrılmış anahtar. Örn: "logging.level"
            default: Anahtar bulunamazsa döndürülecek varsayılan değer.

        Returns:
            Konfigürasyon değeri.

        Kullanım:
            config.get("logging.level")        → "INFO"
            config.get("vision.image_size")    → 224
            config.get("nonexistent", "N/A")   → "N/A"
        """
        keys = key.split(".")
        current: Any = self._merged
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Bir konfigürasyon bölümünün tamamını döndürür.

        Args:
            section: Bölüm adı. Örn: "logging", "vision"

        Returns:
            Bölümün dictionary'si veya boş dict.
        """
        return self._merged.get(section, {})

    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Belirli bir konfigürasyon dosyasının içeriğini döndürür.

        Args:
            config_name: "main", "model" veya "data"

        Returns:
            İlgili konfigürasyon dictionary'si.
        """
        return self._configs.get(config_name, {})

    @property
    def all(self) -> Dict[str, Any]:
        """Tüm birleştirilmiş konfigürasyonu döndürür."""
        return self._merged.copy()

    def reload(self) -> None:
        """Konfigürasyonları diskten yeniden yükler."""
        self._load_all()

    def __repr__(self) -> str:
        sections = list(self._merged.keys())
        return f"Config(sections={sections})"


# ---- Kolaylık fonksiyonları ----

def load_config() -> Config:
    """
    Global konfigürasyon instance'ını döndürür.

    Returns:
        Config singleton instance.
    """
    return Config()


def get_config(section: Optional[str] = None) -> Any:
    """
    Konfigürasyon değerini veya bölümünü döndürür.

    Args:
        section: Bölüm adı (opsiyonel). None ise tüm config döner.

    Returns:
        Konfigürasyon dictionary'si veya tüm config.
    """
    config = Config()
    if section is None:
        return config.all
    return config.get_section(section)
