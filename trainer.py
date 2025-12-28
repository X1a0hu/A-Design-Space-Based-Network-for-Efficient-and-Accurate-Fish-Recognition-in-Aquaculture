import sys
from pathlib import Path
from tqdm import tqdm
import torch
import csv
from typing import Union, Dict
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn

sys.path.append(str(Path(__file__).parent))

from losses import LargeMarginCosineLoss
from estimator import Estimator


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        criterion: LargeMarginCosineLoss,
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler.CosineAnnealingLR,
        save_dir: Union[str, Path],
        device: torch.device = torch.device("cuda"),
        epochs: int = 100,
        interval: int = 5,
    ):
        """
        Args:
            model (nn.Module): SHOUNet model instance
            criterion (nn.Module): Large margin cosine loss function
            optimizer (torch.optim.Optimizer): Optimizer instance
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
            save_dir (Path): Directory to save checkpoints
            device (str): Training device ('cuda' or 'cpu')
            epochs (int): Total training epochs
            interval (int): Checkpoint saving interval
        """
        # Hardware setup
        self.device = device

        # Model components
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training parameters
        self.epochs = epochs
        self.interval = interval
        self.save_dir = Path(save_dir)
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Load Estimator
        self.estimator = self._load_estimator()

        # Metrics tracking
        self.metrics: Dict[str:float] = {
            "train_loss": 0.0,
            "val_loss": 0.0,
            "acc": 0.0,
            "rank1": 0.0,
            "tar_far": 0.0,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.best_metric = float("inf")
        self._setup_logging()

        for epoch in range(self.epochs):
            self._train_epoch(train_loader, epoch)
            self._validate_epoch(val_loader)
            self._update_scheduler()
            self._save_checkpoints(epoch)
            self._log_metrics()

    def _load_estimator(self):
        return Estimator(save_dir=self.save_dir, device=self.device)

    def _setup_logging(self) -> None:
        """Initialize CSV logger."""
        self.csv_file = self.save_dir / "results.csv"
        self.fieldnames = [
            "Train Loss",
            "Validation Loss",
            "Validation Accuracy",
            "Validation Rank-1 Accuracy",
            "Validation TAR@FAR=1e-6",
        ]
        with open(self.csv_file, "w") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def _train_epoch(self, loader: DataLoader, epoch: int) -> None:
        """Single training epoch."""
        self.model.train()
        self.metrics.update({"train_loss": 0.0})

        print("Epoch".rjust(10) + "Loss".rjust(10) + "Precision".rjust(10))
        pbar = tqdm(
            loader,
            bar_format="{desc}{percentage:3.0f}% {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            ncols=80,
            leave=True,
        )

        for images, targets in pbar:
            if not isinstance(images, torch.Tensor):
                raise TypeError(
                    f"Expected images to be torch.Tensor, but got type {type(images)}"
                )
            if not isinstance(targets, torch.Tensor):
                raise TypeError(
                    f"Expected targets to be torch.Tensor, but got type {type(targets)}"
                )

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Reset grad
            self.optimizer.zero_grad()

            # Forward transmission
            embeddings = self.model(images)
            loss = self.criterion(embeddings, targets)

            if not isinstance(loss, torch.Tensor):
                raise TypeError(
                    f"Expected loss to be torch.Tensor, but got type {type(loss)}"
                )

            # Backward transmission
            loss.backward()
            self.optimizer.step()

            # Update metrics
            self.metrics["train_loss"] += loss.item()

            pbar.set_description(
                f"{epoch + 1}/{self.epochs}".rjust(10) + f"{loss.item():.4f}".rjust(10)
            )

        # Calculate epoch metrics
        self.metrics["train_loss"] /= len(loader)

    def _validate_epoch(self, val_loader: DataLoader) -> None:
        """Validation epoch."""
        self.model.eval()
        self.metrics.update({"val_loss": 0.0})

        with torch.no_grad():
            for images, targets in val_loader:
                if not isinstance(images, torch.Tensor):
                    raise TypeError(
                        f"Expected images to be torch.Tensor, but got type {type(images)}"
                    )
                if not isinstance(targets, torch.Tensor):
                    raise TypeError(
                        f"Expected targets to be torch.Tensor, but got type {type(targets)}"
                    )
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                embeddings = self.model(images)
                loss = self.criterion(embeddings, targets)

                if not isinstance(loss, torch.Tensor):
                    raise TypeError(
                        f"Expected loss to be torch.Tensor, but got type {type(loss)}"
                    )

                self.metrics["val_loss"] += loss.item()

        # Calculate validation metrics
        self.metrics["val_loss"] /= len(val_loader)
        self._estimate(loader=val_loader)

    def _update_scheduler(self) -> None:
        """Update learning rate scheduler."""
        prev_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()
        cur_lr = self.scheduler.get_last_lr()[0]

        if cur_lr != prev_lr:
            print(f"Current learning rate: {cur_lr:.6f}")

    def _estimate(self, loader: DataLoader):
        """Estimate model performance on a dataset."""
        results = self.estimator.estimate(model=self.model, dataloader=loader)
        self.metrics.update({"acc": results["acc"]})
        self.metrics.update({"rank1": results["rank_1"]})
        self.metrics.update({"tar_far": results["tar_far"]})

    def _save_checkpoints(self, epoch: int) -> None:
        """Save training checkpoints."""
        checkpoint = {
            "epoch": epoch,
            "weights": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        # Save best and last
        if self.metrics["val_loss"] < self.best_metric:
            torch.save(checkpoint, self.save_dir / "weights" / "best.pt")
            self.best_metric = self.metrics["val_loss"]

        if epoch % self.interval == 0 or epoch == self.epochs - 1:
            torch.save(checkpoint, self.save_dir / "weights" / "last.pt")

    def _log_metrics(self) -> None:
        """Log metrics to CSV and update progress."""
        # Print log
        print(
            f"Train Loss: {self.metrics['train_loss']:.4f}, "
            f"Validation Loss: {self.metrics['val_loss']:.4f}, "
            f"Accuracy(>0.95): {self.metrics['acc']:.4f}, "
            f"Rank-1 Accuracy: {self.metrics['rank1']:.4f}, "
            f"TAR@FAR=1e-6: {self.metrics['tar_far']:.4f}"
        )

        # Write to CSV
        with open(self.csv_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(
                {
                    self.fieldnames[0]: round(self.metrics["train_loss"], 4),
                    self.fieldnames[1]: round(self.metrics["val_loss"], 4),
                    self.fieldnames[2]: round(self.metrics["acc"], 4),
                    self.fieldnames[3]: round(self.metrics["rank1"], 4),
                    self.fieldnames[4]: round(self.metrics["tar_far"], 4),
                }
            )
