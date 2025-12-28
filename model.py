import sys
from pathlib import Path
from typing import Optional, Union, Dict, Set
import yaml
import numpy as np
import torch
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).parent))

from losses import LargeMarginCosineLoss
from dataset import FishDataset
from networks import SHOUNet
from seeknet import SeekNet
from trainer import Trainer


class SHOUModel:

    def __init__(
        self,
        type: str,
        config: Union[str, Path, Dict] = None,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            cfg: Path to config file or config dict
            model_path: Path to the model weights file
        """

        self.type = type

        # Check device
        self.cuda = torch.cuda.is_available()
        self.device: torch.device = torch.device("cuda" if self.cuda else "cpu")

        # Set save path
        self.root_dir: Union[str, Path] = Path(__file__).parent.parent
        self.save_dir: Union[str, Path] = self._save_dir()

        # Data Container
        self.names: Set[str] = []
        self.transform: transforms.Compose = []

        # Load configuration
        self.config: Dict = self._load_config(config)

        # Model setup
        self.model: nn.Module = self._init_model()
        self._setup_transforms()

        # Model setup
        if model_path:
            self.load(model_path)

    def _load_config(self, config: Union[str, Path, Dict]) -> Dict:
        """
        Load configuration from a file or dictionary.
        Args:
            config: Path to config file or config dict
        Returns:
            Configuration dictionary
        """
        if config is None:
            config = Path.cwd() / "cfg" / "models" / "shounet.yaml"

        if isinstance(config, (str, Path)):
            with open(config) as f:
                return yaml.safe_load(f)
        return config

    def _init_model(self) -> nn.Module:
        """
        Initialize model
        """
        return SeekNet(
            stage_type=self.type,
            block_widths=self.config["block_widths"],
            num_blocks=self.config["num_blocks"],
            bottleneck_ratios=self.config["bottleneck_ratios"],
            group_widths=self.config["group_widths"],
            se_ratios=self.config["se_ratios"],
            embed_dim=self.config["embed_dim"],
        )

    def _setup_transforms(self):
        """Configure data augmentations"""
        self.transform = {
            "train": self._get_transforms(train=True),
            "val": self._get_transforms(train=False),
        }

    def _get_transforms(self, train: bool) -> torch.nn.Sequential:
        """Base image transforms"""
        T = [
            transforms.Resize((224, 224)),
        ]
        if train:
            T.extend(
                [
                    transforms.RandomAffine(degrees=20),
                ]
            )
        T.extend(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
            ]
        )
        return transforms.Compose(T)

    def _change_dir(self, dir: Union[str, Path]):
        self.save_dir = dir

    def _save_dir(self):
        save_dir = (
            self.root_dir
            / "runs"
            / "identify"
            # / "samples"
            # / "train"
            / "seek"
            / "train"
        )
        index = 1

        while True:
            if not save_dir.exists():
                break
            save_dir = save_dir.parent / f"train{index}"
            index += 1

        return save_dir

    def _mksvdir(self):
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def load(self, model_path: Union[str, Path]):
        """Load model weights"""
        torch.serialization.add_safe_globals([set])

        ckpt = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(ckpt["weights"])

        self.model.to(self.device)

    def train(
        self,
        data: Union[str, Dict],
        epochs: int = 100,
        device: torch.device = torch.device("cuda"),
        workers: int = 1,
        batch: int = 64,
        lr0: float = 1e-2,
        lrf: float = 1e-2,
    ):
        self._mksvdir()

        # Update config
        self.device = device

        # Load Dataset
        train_dataset = FishDataset(
            data, transform=self.transform["train"], mode="train"
        )
        val_dataset = FishDataset(data, transform=self.transform["val"], mode="val")

        # Load DataLoder
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            drop_last=False,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch,
            shuffle=False,
            num_workers=workers,
            drop_last=False,
        )

        # Check model head
        self.model.to(device)

        self.names = train_dataset.names
        self.config["num_classes"] = len(self.names)

        # Load model, loss, optimizer, and scheduler
        criterion = LargeMarginCosineLoss(
            self.config["embed_dim"],
            self.config["num_classes"],
            self.config["m"],
            self.config["s"],
        ).to(device)

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr0,
            momentum=0.9,
            fused=self.cuda,
            weight_decay=1e-4,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr0 * lrf
        )

        trainer = Trainer(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=self.save_dir,
            device=self.device,
            epochs=epochs,
            interval=5,
        )
        trainer.train(train_loader, val_loader)

    def predict(self, source: Union[str, Path, Image.Image, np.ndarray]):
        """Predict species from an image."""
        if isinstance(source, (str, Path)):
            source = Image.open(source)
        elif isinstance(source, np.ndarray):
            source = Image.fromarray(source)

        if source.mode != "RGB":
            source = source.convert("RGB")

        source = self.transform["val"](source).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embeddings = self.model(source)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    # def mapping(self, image: str, replace=True):
    #     """
    #     Args:
    #         image: A list of PIL image(s) from about the name
    #         name: the real name of the identity
    #     """
    #     embed_dir = Path(self.save_dir) / "embeddings"
    #     embed_dir.mkdir(parents=True, exist_ok=True)
    #     output_filename = Path(image).stem

    #     embed = self.predict(image)[0].cpu().numpy()

    #     output_path = embed_dir / f"{output_filename}.npz"

    #     if not replace and output_path.exists():
    #         print(f"File {output_filename}.npz already exists. Skipping save.")
    #         return

    #     np.savez(output_path, embed=embed, name=output_filename)

    def plot_embeddings(self, data: Union[str, Dict], mode: str = "all", dim: int = 2):
        from sklearn.manifold import TSNE
        from torch.utils.data import ConcatDataset
        import matplotlib.pyplot as plt

        assert mode in [
            "train",
            "val",
            "all",
        ], "Parameter 'mode' must be 'train', 'val' or 'all'."

        if mode == "all":
            dataset = ConcatDataset(
                [
                    FishDataset(data, self.transform["val"], mode="train"),
                    FishDataset(data, self.transform["val"], mode="val"),
                ]
            )
        elif mode == "train":
            dataset = FishDataset(data, self.transform["val"], mode="train")
        else:
            dataset = FishDataset(data, self.transform["val"], mode="val")

        dataloder = DataLoader(
            dataset=dataset, num_workers=1, shuffle=False, drop_last=False
        )

        views = []
        labels = []

        for images, targets in dataloder:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            _, _, view = self.model.predict(images)

            views.append(view.cpu().numpy())
            labels.append(targets.cpu().numpy())

        views = np.concatenate(views, axis=0)
        labels = np.concatenate(labels, axis=0)

        plt.figure(figsize=(10, 10))

        plt.scatter(views[:, 0], views[:, 1], c=labels, cmap="tab20", alpha=0.7)
        plt.show()

        # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        # reduced_features = tsne.fit_transform(features)

        # fig = plt.figure(figsize=(10, 8))
        # if dim == 2:
        #     scatter = plt.scatter(
        #         reduced_features[:, 0],
        #         reduced_features[:, 1],
        #         c=labels,
        #         cmap="tab20",
        #         alpha=0.7,
        #         edgecolors=["black" if idx == 1 else "none" for idx in indices],
        #         linewidths=[1.5 if idx == 1 else 0 for idx in indices],
        #     )
        #     plt.colorbar(scatter, label="Categories")
        #     plt.title("t-SNE Visualization of Embeddings (2D)")
        #     plt.xlabel("t-SNE Component 1")
        #     plt.ylabel("t-SNE Component 2")
        #     plt.show()
        # elif dim == 3:
        #     ax = fig.add_subplot(111, projection="3d")
        #     scatter = ax.scatter(
        #         reduced_features[:, 0],
        #         reduced_features[:, 1],
        #         reduced_features[:, 2],
        #         c=labels,
        #         cmap="tab20",
        #         alpha=0.7,
        #         edgecolors=["black" if idx == 1 else "none" for idx in indices],
        #         linewidths=[1.5 if idx == 1 else 0 for idx in indices],
        #     )
        #     fig.colorbar(scatter)
        #     ax.set_title("t-SNE Visualization of Embeddings (3D)")
        #     ax.set_xlabel("t-SNE Component 1")
        #     ax.set_ylabel("t-SNE Component 2")
        #     ax.set_zlabel("t-SNE Component 3")
        #     plt.show()
        # else:
        #     raise ValueError(f"dim should be 2 or 3 but current dim={dim}")
