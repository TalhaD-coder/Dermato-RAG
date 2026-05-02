"""
Dermato-RAG Veri Augmentation Modulu.

Dermatolojik goruntuler icin bilimsel olarak dogrulanmis
augmentation stratejileri.

Stratejiler:
- Geometrik: Dondurme, cevirme, kirpma
- Renk: Parlaklik, kontrast, doygunluk
- Dermatoloji-spesifik: Sac artefakti, dermoskop cercevesi

Referanslar:
- Perez et al. (2018) "Data Augmentation for Skin Lesion Analysis"
- Barata et al. (2021) "Improving Dermoscopy Image Classification"

Kullanim:
    from src.data.augmentation import (
        get_train_transforms,
        get_eval_transforms,
        get_augmentation_config,
    )
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ImageNet normalizasyon degerleri
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation seviyeleri
AUGMENTATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "light": {
        "random_crop_scale": (0.9, 1.0),
        "horizontal_flip_p": 0.5,
        "vertical_flip_p": 0.0,
        "rotation_degrees": 10,
        "color_jitter": {
            "brightness": 0.1, "contrast": 0.1,
            "saturation": 0.1, "hue": 0.02,
        },
        "random_erasing_p": 0.0,
        "gaussian_blur_p": 0.0,
        "random_affine": False,
    },
    "medium": {
        "random_crop_scale": (0.8, 1.0),
        "horizontal_flip_p": 0.5,
        "vertical_flip_p": 0.5,
        "rotation_degrees": 20,
        "color_jitter": {
            "brightness": 0.2, "contrast": 0.2,
            "saturation": 0.2, "hue": 0.1,
        },
        "random_erasing_p": 0.1,
        "gaussian_blur_p": 0.1,
        "random_affine": True,
    },
    "heavy": {
        "random_crop_scale": (0.7, 1.0),
        "horizontal_flip_p": 0.5,
        "vertical_flip_p": 0.5,
        "rotation_degrees": 30,
        "color_jitter": {
            "brightness": 0.3, "contrast": 0.3,
            "saturation": 0.3, "hue": 0.15,
        },
        "random_erasing_p": 0.2,
        "gaussian_blur_p": 0.2,
        "random_affine": True,
    },
}


def get_augmentation_config(level: str = "medium") -> Dict[str, Any]:
    """Augmentation konfigurasyonunu dondurur."""
    if level not in AUGMENTATION_CONFIGS:
        raise ValueError(
            f"Gecersiz augmentation level: {level}. "
            f"Secenekler: {list(AUGMENTATION_CONFIGS.keys())}"
        )
    return AUGMENTATION_CONFIGS[level].copy()


def get_train_transforms(image_size: int = 224, level: str = "medium") -> Any:
    """
    Egitim icin augmentation transform pipeline olusturur.

    Args:
        image_size: Cikti goruntu boyutu.
        level: Augmentation seviyesi ("light", "medium", "heavy").

    Returns:
        torchvision.transforms.Compose instance.
    """
    try:
        from torchvision import transforms
    except ImportError:
        logger.warning("torchvision yuklu degil")
        return None

    config = get_augmentation_config(level)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    transform_list = []

    # Geometrik donusumler
    transform_list.append(
        transforms.RandomResizedCrop(
            image_size,
            scale=config["random_crop_scale"],
            ratio=(0.9, 1.1),
        )
    )

    if config["horizontal_flip_p"] > 0:
        transform_list.append(
            transforms.RandomHorizontalFlip(p=config["horizontal_flip_p"])
        )
    if config["vertical_flip_p"] > 0:
        transform_list.append(
            transforms.RandomVerticalFlip(p=config["vertical_flip_p"])
        )
    if config["rotation_degrees"] > 0:
        transform_list.append(
            transforms.RandomRotation(degrees=config["rotation_degrees"], fill=0)
        )
    if config.get("random_affine", False):
        transform_list.append(
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05),
                scale=(0.95, 1.05), fill=0,
            )
        )

    # Renk donusumleri
    cj = config["color_jitter"]
    transform_list.append(
        transforms.ColorJitter(
            brightness=cj["brightness"], contrast=cj["contrast"],
            saturation=cj["saturation"], hue=cj["hue"],
        )
    )

    # Gaussian blur
    if config["gaussian_blur_p"] > 0:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=config["gaussian_blur_p"],
            )
        )

    # Tensor + normalizasyon
    transform_list.append(transforms.ToTensor())
    transform_list.append(normalize)

    # Random erasing (tensor uzerinde calisir)
    if config["random_erasing_p"] > 0:
        transform_list.append(
            transforms.RandomErasing(
                p=config["random_erasing_p"],
                scale=(0.02, 0.1), ratio=(0.3, 3.3),
                value="random",
            )
        )

    logger.info(f"Train transforms olusturuldu (level={level}, {len(transform_list)} adim)")
    return transforms.Compose(transform_list)


def get_eval_transforms(image_size: int = 224) -> Any:
    """
    Degerlendirme (val/test) icin transform pipeline.
    Augmentation yok, sadece resize + normalizasyon.
    """
    try:
        from torchvision import transforms
    except ImportError:
        logger.warning("torchvision yuklu degil")
        return None

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


class HairAugmentation:
    """
    Dermoskopik goruntulerdeki sac artefaktini simule eder.

    Rastgele ince cizgiler cizerek modelin sac artefaktlarina
    dayanikli olmasini saglar. (Perez et al. 2018)

    Args:
        p: Uygulanma olasiligi.
        num_hairs_range: Min-max sac sayisi.
        hair_width_range: Min-max sac kalinligi (piksel).
        hair_color_range: Sac rengi araligi (0-255).
    """

    def __init__(
        self,
        p: float = 0.3,
        num_hairs_range: Tuple[int, int] = (3, 10),
        hair_width_range: Tuple[int, int] = (1, 3),
        hair_color_range: Tuple[int, int] = (0, 80),
    ) -> None:
        self.p = p
        self.num_hairs_range = num_hairs_range
        self.hair_width_range = hair_width_range
        self.hair_color_range = hair_color_range

    def __call__(self, img: Image.Image) -> Image.Image:
        """Goruntiye rastgele sac artefakti ekler."""
        if np.random.random() > self.p:
            return img

        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        num_hairs = np.random.randint(
            self.num_hairs_range[0], self.num_hairs_range[1] + 1
        )

        for _ in range(num_hairs):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            angle = np.random.uniform(0, np.pi)
            length = np.random.randint(min(w, h) // 4, min(w, h) // 2)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            gray = np.random.randint(self.hair_color_range[0], self.hair_color_range[1])
            width = np.random.randint(self.hair_width_range[0], self.hair_width_range[1] + 1)
            draw.line([(x1, y1), (x2, y2)], fill=(gray, gray, gray), width=width)

        return img

    def __repr__(self) -> str:
        return f"HairAugmentation(p={self.p}, num_hairs={self.num_hairs_range})"


class MicroscopeAugmentation:
    """
    Dermoskop dairesel cerceve (vignette) efektini simule eder.

    Args:
        p: Uygulanma olasiligi.
        radius_ratio_range: Daire yaricapi orani.
    """

    def __init__(
        self,
        p: float = 0.3,
        radius_ratio_range: Tuple[float, float] = (0.85, 0.95),
    ) -> None:
        self.p = p
        self.radius_ratio_range = radius_ratio_range

    def __call__(self, img: Image.Image) -> Image.Image:
        """Goruntiye dairesel vignette efekti uygular."""
        if np.random.random() > self.p:
            return img

        img = img.copy()
        w, h = img.size
        cx, cy = w // 2, h // 2
        ratio = np.random.uniform(self.radius_ratio_range[0], self.radius_ratio_range[1])
        radius = int(min(w, h) // 2 * ratio)

        mask = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius], fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))

        black_bg = Image.new("RGB", (w, h), (0, 0, 0))
        return Image.composite(img, black_bg, mask)

    def __repr__(self) -> str:
        return f"MicroscopeAugmentation(p={self.p}, radius_ratio={self.radius_ratio_range})"


def get_advanced_train_transforms(
    image_size: int = 224,
    level: str = "heavy",
    include_hair: bool = True,
    include_microscope: bool = True,
) -> Any:
    """
    Dermatoloji-spesifik ileri augmentation pipeline.

    Standart torchvision transformlarina ek olarak sac artefakti
    ve dermoskop cercevesi simule eder.
    """
    try:
        from torchvision import transforms
    except ImportError:
        logger.warning("torchvision yuklu degil")
        return None

    config = get_augmentation_config(level)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    transform_list = []

    # PIL-level dermatoloji augmentasyonlari (ToTensor oncesi)
    if include_hair:
        transform_list.append(HairAugmentation(p=0.3))
    if include_microscope:
        transform_list.append(MicroscopeAugmentation(p=0.2))

    # Geometrik
    transform_list.append(
        transforms.RandomResizedCrop(image_size, scale=config["random_crop_scale"])
    )
    if config["horizontal_flip_p"] > 0:
        transform_list.append(transforms.RandomHorizontalFlip(p=config["horizontal_flip_p"]))
    if config["vertical_flip_p"] > 0:
        transform_list.append(transforms.RandomVerticalFlip(p=config["vertical_flip_p"]))
    if config["rotation_degrees"] > 0:
        transform_list.append(transforms.RandomRotation(degrees=config["rotation_degrees"]))

    # Renk
    cj = config["color_jitter"]
    transform_list.append(
        transforms.ColorJitter(
            brightness=cj["brightness"], contrast=cj["contrast"],
            saturation=cj["saturation"], hue=cj["hue"],
        )
    )

    # Blur
    if config["gaussian_blur_p"] > 0:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3)],
                p=config["gaussian_blur_p"],
            )
        )

    # Tensor + normalizasyon
    transform_list.append(transforms.ToTensor())
    transform_list.append(normalize)

    # Random erasing
    if config["random_erasing_p"] > 0:
        transform_list.append(
            transforms.RandomErasing(p=config["random_erasing_p"], scale=(0.02, 0.1), value="random")
        )

    logger.info(f"Advanced transforms olusturuldu (level={level}, hair={include_hair}, microscope={include_microscope})")
    return transforms.Compose(transform_list)


def denormalize(tensor: Any, mean: List[float] = None, std: List[float] = None) -> Any:
    """
    Normalize edilmis tensoru orijinal piksel degerlerine cevirir.
    Gorsellestirme icin kullanilir.
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    try:
        import torch
    except ImportError:
        return tensor

    if not isinstance(tensor, torch.Tensor):
        return tensor

    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp_(0, 1)
