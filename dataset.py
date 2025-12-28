import sys
from pathlib import Path
from typing import Union, Dict, List, Set
from torchvision import transforms
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).parent))
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FishDataset(Dataset):

    def __init__(
        self, data: Union[str, Path], transform: transforms.Compose, mode: str = "train"
    ):
        """
        Arg:
            data: the absolute path to the dataset configuration file (YAML format).
            transform: a torchvision.transforms object to apply to the images.
            mode: the mode in which the dataset is used, either 'train' or 'val'. Default is 'train'.
        """
        assert mode in [
            "train",
            "val",
            "test",
        ], "The dataset type must be 'train', 'val' or 'test'."

        super(FishDataset, self).__init__
        self.transform = transform
        self.mode = mode.lower()

        # Initialize paths
        self.config_path = Path(data).resolve()

        # Data container
        self.images: List[Path] = []
        self.labels: List[int] = []
        self.names: Set[str] = []
        self.nc: int = 0

        self.image_dir: Path = ""
        self.label_dir: Path = ""

        # Load config and data
        self._load_config()
        self._set_directories()
        self._load_samples()

    def _load_config(self) -> Dict:
        """Load dataset configuration from YAML file."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        self.dataset_root = Path(config.get("path")).resolve()
        self._image_subdir = {
            "train": config.get("train", "images/train"),
            "val": config.get("val", "images/val"),
        }
        self._label_subdir = {"train": "labels/train", "val": "labels/val"}

        self.names = self._parse_names(config.get("names"))
        self.nc = config.get("nc", 0)

    def _parse_names(self, names: Union[Dict, List]) -> List[str]:
        """Parse class names from YAML."""
        if isinstance(names, dict):
            names = list(names.values())
        return set(names)

    def _set_directories(self):
        """Set image and label directories based on mode."""
        self.image_dir = self.dataset_root / self._image_subdir[self.mode]
        self.label_dir = self.dataset_root / self._label_subdir[self.mode]

        # Validate paths
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

    def _load_samples(self):
        valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}

        for image_path in self.image_dir.iterdir():
            if image_path.suffix.lower() not in valid_suffixes:
                continue

            label_path = (self.label_dir / image_path.stem).with_suffix(".txt")
            if not label_path.exists():
                continue

            with label_path.open("r") as f:
                try:
                    label = int(f.read().strip())
                    self.images.append(image_path)
                    self.labels.append(label)
                except ValueError:
                    continue

    def __getitem__(self, index):
        data = Image.open(self.images[index]).convert("RGB")
        target = self.labels[index]
        data = self.transform(data)

        return data, target

    def __len__(self):
        return len(self.labels)
