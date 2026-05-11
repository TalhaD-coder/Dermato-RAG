"""
Dermato-RAG Model Eğitim Modülü.

Vision encoder modelinin fine-tuning eğitim döngüsünü yönetir.
- Train/Validation döngüsü
- Early stopping
- Checkpoint kaydetme
- Learning rate scheduling
- Mixed precision training (fp16)

Kullanım:
    from src.models.trainer import VisionTrainer
    from src.models.vision_encoder import DermatoVisionEncoder

    model = DermatoVisionEncoder(num_classes=9, mode="classify")
    trainer = VisionTrainer(model, train_loader, val_loader)
    trainer.train(num_epochs=20)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class VisionTrainer:
    """
    Vision encoder eğitim yöneticisi.

    Args:
        model: DermatoVisionEncoder instance.
        train_loader: Eğitim DataLoader.
        val_loader: Doğrulama DataLoader.
        optimizer: PyTorch optimizer (None ise AdamW kullanılır).
        scheduler: LR scheduler (None ise CosineAnnealing kullanılır).
        criterion: Loss fonksiyonu (None ise CrossEntropy kullanılır).
        device: Eğitim cihazı ('cuda' veya 'cpu').
        output_dir: Checkpoint kayıt dizini.
        class_weights: Sınıf ağırlıkları (imbalanced data için).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        output_dir: str = "outputs/vision",
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        # Cihaz belirleme
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss fonksiyonu (sınıf dengesizliği ağırlıkları ile)
        if criterion is not None:
            self.criterion = criterion
        elif class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4,
                weight_decay=0.01,
            )

        # Scheduler
        self.scheduler = scheduler

        # Eğitim geçmişi
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0

        logger.info(f"Trainer oluşturuldu — Cihaz: {self.device}")

    def train(
        self,
        num_epochs: int = 20,
        early_stopping_patience: int = 5,
        use_amp: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Model eğitim döngüsü.

        Args:
            num_epochs: Toplam epoch sayısı.
            early_stopping_patience: Kaç epoch iyileşme olmazsa dur.
            use_amp: Mixed precision (fp16) kullan.

        Returns:
            Eğitim geçmişi dictionary.
        """
        # Mixed precision scaler
        use_amp = use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # Scheduler yoksa oluştur
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=1e-6
            )

        patience_counter = 0

        logger.info(f"Eğitim başlıyor: {num_epochs} epoch, AMP={use_amp}")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_acc = self._train_epoch(epoch, scaler, use_amp)

            # Validate
            val_loss, val_acc = self._validate_epoch(epoch)

            # LR step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()

            # Geçmişe kaydet
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - start_time

            logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
                f"LR: {current_lr:.2e} | {elapsed:.1f}s"
            )

            # Best model kaydet
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                self._save_best(epoch, val_acc, val_loss)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping: {early_stopping_patience} epoch boyunca "
                    f"iyileşme yok. En iyi epoch: {self.best_epoch} "
                    f"(acc={self.best_val_acc:.2%})"
                )
                break

        logger.info(
            f"Eğitim tamamlandı! En iyi: epoch={self.best_epoch}, "
            f"val_acc={self.best_val_acc:.2%}"
        )
        return self.history

    def _train_epoch(
        self, epoch: int, scaler: Any, use_amp: bool
    ) -> Tuple[float, float]:
        """Tek bir eğitim epoch'u."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def _validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """Tek bir doğrulama epoch'u."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def _save_best(self, epoch: int, val_acc: float, val_loss: float) -> None:
        """En iyi modeli kaydeder."""
        save_path = self.output_dir / "best_model.pt"
        self.model.save_checkpoint(
            str(save_path),
            epoch=epoch,
            metrics={"val_acc": val_acc, "val_loss": val_loss},
        )
        logger.info(f"✅ En iyi model kaydedildi: val_acc={val_acc:.2%}")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test seti üzerinde değerlendirme.

        Args:
            test_loader: Test DataLoader.

        Returns:
            Metrik sonuçları.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        accuracy = correct / total
        avg_loss = total_loss / total

        results = {
            "test_loss": avg_loss,
            "test_acc": accuracy,
            "total_samples": total,
            "correct": correct,
        }

        logger.info(
            f"Test sonuçları: Loss={avg_loss:.4f}, "
            f"Acc={accuracy:.2%} ({correct}/{total})"
        )
        return results
