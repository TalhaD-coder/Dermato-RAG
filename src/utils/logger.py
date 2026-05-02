"""
Dermato-RAG Loglama Altyapısı.

Rich kütüphanesi ile renkli konsol çıktısı ve dosya loglama
desteği sağlar. Yapılandırılmış (structured) loglama formatı
ile deneyler ve hata ayıklama kolaylaştırılır.

Kullanım:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Model yüklendi", extra={"model": "BiomedCLIP"})
    logger.warning("GPU bulunamadı, CPU kullanılacak")
    logger.error("Veri dosyası bulunamadı", exc_info=True)
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Dermato-RAG özel tema
_DERMATO_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "debug": "dim",
        "success": "bold green",
        "module": "bold magenta",
        "timestamp": "dim cyan",
    }
)

# Global console instance
console = Console(theme=_DERMATO_THEME)

# Oluşturulan logger'ları takip et (duplikasyonu önle)
_initialized_loggers: dict[str, logging.Logger] = {}


class DermatoFormatter(logging.Formatter):
    """
    Dermato-RAG için özel log formatter.

    Dosya logları için yapılandırılmış format sağlar.
    Her log satırı: timestamp | level | module | message
    """

    def format(self, record: logging.LogRecord) -> str:
        # Zaman damgası
        timestamp = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]

        # Modül bilgisi
        module = record.module if record.module else "unknown"

        # Temel format
        log_line = (
            f"{timestamp} | {record.levelname:<8} | {module:<20} | {record.getMessage()}"
        )

        # Exception bilgisi varsa ekle
        if record.exc_info and record.exc_info[0] is not None:
            log_line += f"\n{self.formatException(record.exc_info)}"

        return log_line


def _ensure_log_dir(log_file: str) -> Path:
    """Log dizininin var olduğundan emin ol."""
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Yapılandırılmış logger oluşturur veya mevcut olanı döndürür.

    Konfigürasyon dosyasından varsayılan değerleri okur,
    parametrelerle override edilebilir.

    Args:
        name: Logger adı (genellikle __name__).
        level: Log seviyesi override. None ise config'den alınır.
        log_to_file: Dosyaya log yazılsın mı? None ise config'den alınır.
        log_file: Log dosyası yolu. None ise config'den alınır.

    Returns:
        Yapılandırılmış logging.Logger instance.

    Kullanım:
        logger = get_logger(__name__)
        logger.info("İşlem başladı")
        logger.debug("Detaylı bilgi: %s", data)
        logger.error("Hata oluştu", exc_info=True)
    """
    # Daha önce oluşturulmuş ise döndür
    if name in _initialized_loggers:
        return _initialized_loggers[name]

    # Konfigürasyondan varsayılanları yükle (döngüsel import'u önle)
    log_config = _get_log_config()

    # Parametreleri belirle
    effective_level = level or log_config.get("level", "INFO")
    effective_log_to_file = log_to_file if log_to_file is not None else log_config.get(
        "log_to_file", True
    )
    effective_log_file = log_file or log_config.get("log_file", "logs/dermato_rag.log")

    # Logger oluştur
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, effective_level.upper(), logging.INFO))
    logger.propagate = False  # Üst logger'a yayılmasını önle

    # Mevcut handler'ları temizle
    logger.handlers.clear()

    # ---- Rich Console Handler ----
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        show_level=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
        log_time_format="[%H:%M:%S]",
    )
    rich_handler.setLevel(getattr(logging, effective_level.upper(), logging.INFO))
    logger.addHandler(rich_handler)

    # ---- File Handler (opsiyonel) ----
    if effective_log_to_file:
        try:
            log_path = _ensure_log_dir(effective_log_file)
            max_bytes = log_config.get("max_file_size_mb", 50) * 1024 * 1024
            backup_count = log_config.get("backup_count", 5)

            file_handler = RotatingFileHandler(
                filename=str(log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)  # Dosyaya her şeyi yaz
            file_handler.setFormatter(DermatoFormatter())
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            # Dosya yazılamıyorsa sadece konsola logla
            logger.warning(f"Log dosyası oluşturulamadı: {e}. Sadece konsol kullanılacak.")

    # Cache'e kaydet
    _initialized_loggers[name] = logger

    return logger


def _get_log_config() -> dict:
    """
    Loglama konfigürasyonunu güvenli şekilde yükler.
    Döngüsel import'u önlemek için lazy loading yapar.
    """
    try:
        from src.utils.config import Config
        config = Config()
        return config.get_section("logging")
    except Exception:
        # Config yüklenemezse varsayılanları kullan
        return {
            "level": "INFO",
            "log_to_file": True,
            "log_file": "logs/dermato_rag.log",
            "max_file_size_mb": 50,
            "backup_count": 5,
        }


def log_separator(logger: logging.Logger, title: str = "", char: str = "=", length: int = 60) -> None:
    """
    Log çıktısında görsel ayırıcı oluşturur.

    Args:
        logger: Logger instance.
        title: Ayırıcı başlığı (opsiyonel).
        char: Ayırıcı karakteri.
        length: Ayırıcı uzunluğu.
    """
    if title:
        padding = (length - len(title) - 2) // 2
        separator = f"{char * padding} {title} {char * padding}"
    else:
        separator = char * length
    logger.info(separator)


def log_dict(logger: logging.Logger, data: dict, title: str = "Parametreler") -> None:
    """
    Dictionary'yi okunabilir formatta loglar.

    Args:
        logger: Logger instance.
        data: Loglanacak dictionary.
        title: Başlık.
    """
    log_separator(logger, title)
    for key, value in data.items():
        logger.info(f"  {key:<30}: {value}")
    log_separator(logger)
