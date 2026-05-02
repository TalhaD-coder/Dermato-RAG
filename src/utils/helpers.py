"""
Dermato-RAG Yardımcı Fonksiyonlar.

Proje genelinde kullanılan ortak yardımcı fonksiyonları içerir:
- Seed ayarlama (tekrarlanabilirlik)
- Cihaz (device) belirleme
- Dosya/dizin operasyonları
- Zamanlama araçları
"""

import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def set_seed(seed: int = 42) -> None:
    """
    Global random seed'i ayarlar. Tekrarlanabilirlik için kritik.

    Python random, NumPy ve PyTorch seed'lerini eş zamanlı ayarlar.

    Args:
        seed: Random seed değeri.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(f"PyTorch seed ayarlandı: {seed}")
    except ImportError:
        logger.debug("PyTorch yüklü değil, sadece Python/NumPy seed ayarlandı")

    logger.info(f"Global seed ayarlandı: {seed}")


def get_device(gpu_id: int = 0) -> Any:
    """
    Uygun hesaplama cihazını (GPU/CPU) belirler ve döndürür.

    Args:
        gpu_id: Kullanılacak GPU ID'si.

    Returns:
        torch.device instance.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_mem / (1024**3)
            logger.info(f"GPU kullanılacak: {gpu_name} ({gpu_memory:.1f} GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple MPS kullanılacak")
        else:
            device = torch.device("cpu")
            logger.warning("GPU bulunamadı, CPU kullanılacak")

        return device
    except ImportError:
        logger.warning("PyTorch yüklü değil")
        return None


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Dizinin var olduğundan emin olur, yoksa oluşturur.

    Args:
        path: Oluşturulacak dizin yolu.

    Returns:
        Oluşturulan/mevcut dizinin Path objesi.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Proje kök dizinini döndürür."""
    return PROJECT_ROOT


def resolve_path(relative_path: Union[str, Path]) -> Path:
    """
    Göreceli yolu proje kökünden mutlak yola çevirir.

    Args:
        relative_path: Proje köküne göre yol.

    Returns:
        Mutlak Path objesi.
    """
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@contextmanager
def timer(description: str = "İşlem") -> Generator[None, None, None]:
    """
    Kod bloğunun çalışma süresini ölçen context manager.

    Args:
        description: Zamanlanan işlemin açıklaması.

    Kullanım:
        with timer("Model yükleme"):
            model = load_model()
        # Çıktı: "Model yükleme tamamlandı: 3.45 saniye"
    """
    start_time = time.perf_counter()
    logger.info(f"[START] {description} basladi...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        if elapsed < 60:
            time_str = f"{elapsed:.2f} saniye"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes} dakika {seconds:.1f} saniye"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            time_str = f"{hours} saat {minutes} dakika"
        logger.info(f"[DONE] {description} tamamlandi: {time_str}")


def format_size(size_bytes: int) -> str:
    """
    Byte cinsinden boyutu okunabilir formata çevirir.

    Args:
        size_bytes: Boyut (byte).

    Returns:
        Okunabilir boyut string'i (örn: "1.5 GB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def count_parameters(model: Any) -> dict:
    """
    PyTorch modelinin parametre sayısını hesaplar.

    Args:
        model: PyTorch nn.Module instance.

    Returns:
        Dict with total, trainable, and frozen parameter counts.
    """
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "total_formatted": format_size(total * 4),  # float32 assumed
        }
    except Exception as e:
        logger.error(f"Parametre sayımı başarısız: {e}")
        return {"total": 0, "trainable": 0, "frozen": 0}
