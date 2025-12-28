import sys
from pathlib import Path
import numpy as np
import random
from typing import Dict, List, Any
from fvcore.nn import FlopCountAnalysis
import torch
import yaml
import shutil

sys.path.append(str(Path(__file__).parent))
random.seed(42)
np.random.seed(42)

from seeknet import SeekNet
from model import SHOUModel


class SampleNet:
    def __init__(self):
        self.save_dir = Path.cwd() / "runs" / "identify" / "samples"

        self.NUM_BLOCK = [i for i in range(1, 13)]
        self.BLOCK_WIDTH = [i for i in range(8, 1025, 8)]
        self.BOTTLENECK_RATIO = [1, 2, 4]
        self.GROUP_WIDTH = [1, 2, 4, 8, 16, 32]
        self.SE_RATIO = [4, 8, 12, 16, 20, 24]
        self.stage = 4

        self.min_flops = 360e6
        self.max_flops = 400e6
        # self.max_flops = 1.6e9

    def _log_uniform_sample(self, space: List, num_samples: int) -> List:
        log_space = np.log(space)
        log_samples = np.random.uniform(log_space[0], log_space[-1], num_samples)
        samples = np.exp(log_samples)

        discrete_samples = [
            space[np.abs(space - sample).argmin()] for sample in samples
        ]

        return discrete_samples

    def _valid_sample(self, sample: Dict) -> bool:
        block_widths = sample["block_widths"]
        bottleneck_ratios = sample["bottleneck_ratios"]
        group_widths = sample["group_widths"]
        se_ratios = sample["se_ratios"]

        for i in range(4):
            if i > 0 and block_widths[i] <= block_widths[i - 1]:
                return False

            hidden_channel = block_widths[i] // bottleneck_ratios[i]
            if hidden_channel < 8:
                return False

            if group_widths[i] > hidden_channel:
                return False

            groups = max(hidden_channel // group_widths[i], 1)
            if hidden_channel % groups != 0:
                return False

            if se_ratios[i] > hidden_channel or hidden_channel // se_ratios[i] == 0:
                return False

        if type is not None:
            model = SeekNet(
                stage_type="CD",
                block_widths=sample["block_widths"],
                num_blocks=sample["num_blocks"],
                bottleneck_ratios=sample["bottleneck_ratios"],
                group_widths=sample["group_widths"],
                se_ratios=sample["se_ratios"],
                embed_dim=512,
            ).to("cuda")

            with torch.no_grad():
                flops = FlopCountAnalysis(
                    model, torch.randn(1, 3, 224, 224).to("cuda")
                ).total()
                return self.min_flops <= flops <= self.max_flops

        return True

    def _save_sample(self, sample: Dict, save_dir: Path, sample_id: int):
        sample["embed_dim"] = 512
        sample["m"] = 0.35
        sample["s"] = 32
        file_path = save_dir / f"sample_{sample_id}.yaml"
        with open(file_path, "a") as f:
            yaml.dump(sample, f, default_flow_style=False, allow_unicode=True)
        print(f"Samples saved to {file_path}")


class SampleNetA(SampleNet):

    def __init__(self):
        super(SampleNetA, self).__init__()
        self.save_dir_a = self.save_dir / "sample_a_no"

    def sample_design(
        self, num_samples: int = 200, type: str = "X", save: bool = False
    ):
        save_dir = self.save_dir_a / type
        save_dir.mkdir(parents=True, exist_ok=True)

        cur_samples = 0
        samples = []

        while cur_samples < num_samples:
            sample = self._generate_sample()
            if self._valid_sample(sample, type):
                samples.append(sample)
                cur_samples += 1

                if save:
                    self._save_sample(sample, self.save_dir_a, cur_samples)

        return samples

    def _generate_sample(self) -> Dict:
        num_blocks = self._log_uniform_sample(self.NUM_BLOCK, self.stage)
        block_widths = self._log_uniform_sample(self.BLOCK_WIDTH, self.stage)
        bottleneck_ratios = self._log_uniform_sample(self.BOTTLENECK_RATIO, self.stage)
        group_widths = self._log_uniform_sample(self.GROUP_WIDTH, self.stage)
        se_ratios = self._log_uniform_sample(self.SE_RATIO, self.stage)

        return {
            "block_widths": block_widths,
            "num_blocks": num_blocks,
            "bottleneck_ratios": bottleneck_ratios,
            "group_widths": group_widths,
            "se_ratios": se_ratios,
        }


class SampleNetB(SampleNet):
    def __init__(self):
        super(SampleNetB, self).__init__()
        self.save_dir_b = self.save_dir / "sample_b"

    # def sample_design(self, sources: Union[List[Dict], str, Path], save: bool = False):
    #     self.save_dir_b.mkdir(parents=True, exist_ok=True)

    #     if isinstance(sources, (str, Path)):
    #         sorted_samples = []

    #         sample_dir = sorted(sources.iterdir(), key=lambda x: x.stem)
    #         for sample in sample_dir:
    #             with open(sample, "r") as f:
    #                 sample_data = yaml.safe_load(f)
    #                 sorted_samples.append(sample_data)

    #         sources = sorted_samples

    #     samples = []
    #     for ratio in self.BOTTLENECK_RATIO:
    #         cur_samples = 0
    #         save_dir = self.save_dir_b / f"sample_{ratio}"
    #         save_dir.mkdir(parents=True, exist_ok=True)

    #         for sample in sources:
    #             sample = self._generate_sample(sample, ratio)
    #             if self._valid_sample(sample):
    #                 samples.append(sample)
    #                 cur_samples += 1

    #                 if save:
    #                     self._save_sample(sample, save_dir, cur_samples)

    #     return samples
    def sample_design(
        self, num_samples: int = 100, bottleneck_ratio: int = 1, save: bool = False
    ):
        self.save_dir_b.mkdir(parents=True, exist_ok=True)

        cur_samples = 0
        samples = []

        while cur_samples < num_samples:
            save_dir = self.save_dir_b / f"bottleneck_{bottleneck_ratio}"
            save_dir.mkdir(parents=True, exist_ok=True)

            sample = self._generate_sample(bottleneck_ratio=bottleneck_ratio)
            if self._valid_sample(sample):
                samples.append(sample)
                cur_samples += 1

                if save:
                    self._save_sample(sample, save_dir, cur_samples)

        return samples

    # def _generate_sample(self, sample: Dict, bottleneck_ratio: int) -> Dict:
    #     sample["bottleneck_ratios"] = [bottleneck_ratio] * self.stage
    #     return sample

    def _generate_sample(self, bottleneck_ratio) -> Dict:
        num_blocks = self._log_uniform_sample(self.NUM_BLOCK, self.stage)
        block_widths = self._log_uniform_sample(self.BLOCK_WIDTH, self.stage)
        bottleneck_ratios = [bottleneck_ratio] * 4
        group_widths = self._log_uniform_sample(self.GROUP_WIDTH, self.stage)
        se_ratios = self._log_uniform_sample(self.SE_RATIO, self.stage)

        return {
            "block_widths": block_widths,
            "num_blocks": num_blocks,
            "bottleneck_ratios": bottleneck_ratios,
            "group_widths": group_widths,
            "se_ratios": se_ratios,
        }


class SampleNetC(SampleNet):

    def __init__(self):
        super(SampleNetC, self).__init__()
        self.save_dir_c = self.save_dir / "sample_c"

    def sample_design(
        self, num_samples: int = 100, group_width: int = 1, save: bool = False
    ):
        self.save_dir_c.mkdir(parents=True, exist_ok=True)

        cur_samples = 0
        samples = []

        while cur_samples < num_samples:
            save_dir = self.save_dir_c / f"groupw_{group_width}"
            save_dir.mkdir(parents=True, exist_ok=True)

            sample = self._generate_sample(group_width=group_width)
            if self._valid_sample(sample):
                samples.append(sample)
                cur_samples += 1

                if save:
                    self._save_sample(sample, save_dir, cur_samples)

        return samples

    def _generate_sample(self, group_width) -> Dict:
        num_blocks = self._log_uniform_sample(self.NUM_BLOCK, self.stage)
        block_widths = self._log_uniform_sample(self.BLOCK_WIDTH, self.stage)
        bottleneck_ratios = [2] * 4
        group_widths = [group_width] * 4
        se_ratios = self._log_uniform_sample(self.SE_RATIO, self.stage)

        return {
            "block_widths": block_widths,
            "num_blocks": num_blocks,
            "bottleneck_ratios": bottleneck_ratios,
            "group_widths": group_widths,
            "se_ratios": se_ratios,
        }


class SampleNetD(SampleNet):
    def __init__(self):
        super(SampleNetD, self).__init__()
        self.save_dir_d = self.save_dir / "sample_d"

    def sample_design(
        self, num_samples: int = 100, se_ratio: int = 1, save: bool = False
    ):
        self.save_dir_d.mkdir(parents=True, exist_ok=True)

        cur_samples = 0
        samples = []

        while cur_samples < num_samples:
            save_dir = self.save_dir_d / f"se_{se_ratio}"
            save_dir.mkdir(parents=True, exist_ok=True)

            sample = self._generate_sample(se_ratio=se_ratio)
            if self._valid_sample(sample):
                samples.append(sample)
                cur_samples += 1

                if save:
                    self._save_sample(sample, save_dir, cur_samples)

        return samples

    def _generate_sample(self, se_ratio: int) -> Dict:
        num_blocks = self._log_uniform_sample(self.NUM_BLOCK, self.stage)
        block_widths = self._log_uniform_sample(self.BLOCK_WIDTH, self.stage)
        bottleneck_ratios = [2] * 4
        group_widths = [1] * 4
        se_ratios = [se_ratio] * 4

        return {
            "block_widths": block_widths,
            "num_blocks": num_blocks,
            "bottleneck_ratios": bottleneck_ratios,
            "group_widths": group_widths,
            "se_ratios": se_ratios,
        }


class SeekNetA:
    def __init__(self):
        self.save_dir = Path.cwd() / "runs" / "identify" / "samples" / "seek_a"

        self.TOTAL_BLOCKS = np.arange(4, 33)
        self.BASE_WIDTH = np.arange(8, 257, 8)
        self.WIDTH_INCREMENT = np.arange(1, 129)
        self.MULTIP = np.arange(1.5, 3.1, 0.1)
        self.BIAS = np.arange(-2, 2.1, 0.1)
        self.SMOOTH = np.arange(0.1, 1.1, 0.1)

        self.BOTTLENECK_RATIO = np.array([1, 2, 4])
        self.GROUP_WIDTH = np.array([1, 2, 4, 8, 16, 32])
        self.SE_RATIO = np.array([16, 20, 24])

        self.stage = 4
        # 400e6, 800e6, 1600e6, 3200e6, 6400e6
        self.flops = [
            [396e6, 404e6],
            [792e6, 808e6],
            [1584e6, 1616e6],
            [3168e6, 3232e6],
            [6336e6, 6464e6],
        ]

    def sample_design(self, num_samples: int = 100, save: bool = False):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        cur_samples = 0
        samples = []

        while cur_samples < num_samples:
            raw_sample = self._generate_sample()
            sample = self._convert(raw_sample=raw_sample)

            if sample is not None and self._valid_sample(sample=sample):
                flops = self._get_flops(sample)

                for idx, (min_f, max_f) in enumerate(self.flops):
                    if min_f <= flops <= max_f:
                        sample.update(raw_sample)
                        sample["fflops"] = idx
                        sample["flops"] = flops

                        sample = self._convert_np_to_native(sample)
                        samples.append(sample)
                        cur_samples += 1

                        if save:
                            self._save_sample(sample, self.save_dir, cur_samples)

    def _convert(self, raw_sample: Dict) -> Dict:
        d = raw_sample["total_blocks"]
        wb = raw_sample["base_width"]
        w_delta = raw_sample["width_increment"]
        w_q = raw_sample["multip"]
        w_gamma = raw_sample["bias"]
        w_tau = raw_sample["smooth"]

        j = np.arange(d)

        u_j = wb + w_delta * (1 + w_gamma * self._sigmoid(w_tau * j)) * j
        s_j = np.log(u_j / wb) / np.log(w_q)
        s_j_quantized = np.round(s_j).astype(int)
        s_j_unique, num_blocks = np.unique(s_j_quantized, return_counts=True)

        if not s_j_unique.size == 4:
            return None

        block_widths = np.floor(wb * (w_q**s_j_unique) / 8) * 8
        block_widths = block_widths.astype(int)

        num_blocks = num_blocks
        bottleneck_ratios = [raw_sample["bottleneck_ratio"]] * 4
        group_widths = [raw_sample["group_width"]] * 4
        se_ratios = [raw_sample["se_ratio"]] * 4

        return {
            "block_widths": block_widths,
            "num_blocks": num_blocks,
            "bottleneck_ratios": bottleneck_ratios,
            "group_widths": group_widths,
            "se_ratios": se_ratios,
        }

    def _get_flops(self, sample: Dict):
        model = SeekNet(
            stage_type="CD",
            block_widths=sample["block_widths"],
            num_blocks=sample["num_blocks"],
            bottleneck_ratios=sample["bottleneck_ratios"],
            group_widths=sample["group_widths"],
            se_ratios=sample["se_ratios"],
            embed_dim=512,
        ).to("cuda")

        with torch.no_grad():
            flops = FlopCountAnalysis(
                model, torch.randn(1, 3, 224, 224).to("cuda")
            ).total()

            return flops

    def _generate_sample(self) -> Dict:
        total_blocks = np.random.choice(self.TOTAL_BLOCKS)
        base_width = np.random.choice(self.BASE_WIDTH)
        width_increment = np.random.choice(self.WIDTH_INCREMENT)
        multip = np.random.choice(self.MULTIP)
        bias = np.random.choice(self.BIAS)
        smooth = np.random.choice(self.SMOOTH)

        bottleneck_ratio = np.random.choice(self.BOTTLENECK_RATIO)
        group_width = np.random.choice(self.GROUP_WIDTH)
        se_ratio = np.random.choice(self.SE_RATIO)

        return {
            "total_blocks": total_blocks,
            "base_width": base_width,
            "width_increment": width_increment,
            "multip": multip,
            "bias": bias,
            "smooth": smooth,
            "bottleneck_ratio": bottleneck_ratio,
            "group_width": group_width,
            "se_ratio": se_ratio,
        }

    def _sigmoid(self, x):
        return 1 - 1 / (1 + np.exp(-x))

    def _valid_sample(self, sample: Dict) -> bool:
        block_widths = sample["block_widths"]
        bottleneck_ratios = sample["bottleneck_ratios"]
        group_widths = sample["group_widths"]
        se_ratios = sample["se_ratios"]

        for i in range(4):
            if i > 0 and block_widths[i] <= block_widths[i - 1]:
                return False

            hidden_channel = block_widths[i] // bottleneck_ratios[i]
            if hidden_channel < 8:
                return False

            if group_widths[i] > hidden_channel:
                return False

            groups = max(hidden_channel // group_widths[i], 1)
            if hidden_channel % groups != 0:
                return False

            if se_ratios[i] > hidden_channel or hidden_channel // se_ratios[i] == 0:
                return False

        return True

    def _save_sample(self, sample: Dict, save_dir: Path, sample_id: int):
        sample["embed_dim"] = 512
        sample["m"] = 0.35
        sample["s"] = 32
        file_path = save_dir / f"sample_{sample_id}.yaml"
        with open(file_path, "a") as f:
            yaml.dump(sample, f, default_flow_style=False, allow_unicode=True)
        print(f"Samples saved to {file_path}")

    def _convert_np_to_native(self, obj):

        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.item() if obj.size == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_np_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_np_to_native(v) for v in obj]
        else:
            return obj


class SeekNetB(SeekNetA):
    def __init__(self):
        super(SeekNetA, self).__init__()
        self.save_dir = Path.cwd() / "runs" / "identify" / "samples" / "seek_b_3"

        self.TOTAL_BLOCKS = np.arange(5, 21)
        self.BASE_WIDTH = np.arange(48, 129, 8)
        self.WIDTH_INCREMENT = np.arange(64, 121, 4)
        self.MULTIP = np.arange(1.8, 2.9, 0.1)
        self.BIAS = np.arange(0, 2.1, 0.1)
        self.SMOOTH = np.arange(0.1, 0.9, 0.1)

        self.BOTTLENECK_RATIO = [2]
        self.GROUP_WIDTH = [1]
        self.SE_RATIO = [16]

        self.stage = 4
        self.flops = [
            [280e6, 520e6],
            [560e6, 1040e6],
            [1120e6, 2080e6],
            [2240e6, 4160e6],
            [4480e6, 8320e6],
        ]

    def sample_design(self, num_samples: int = 100, save: bool = False):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        cur_samples = 0
        samples = []
        samples_per_range = num_samples // len(self.flops)

        for idx, target_flops in enumerate(self.flops):
            target_min = target_flops[0]
            target_max = target_flops[1]

            while cur_samples < samples_per_range * (idx + 1):
                raw_sample = self._generate_sample()
                sample = self._convert(raw_sample=raw_sample)

                if sample is not None and self._valid_sample(sample=sample):
                    flops = self._get_flops(sample)

                    if target_min <= flops <= target_max:
                        sample.update(raw_sample)
                        sample["flops"] = flops

                        sample = self._convert_np_to_native(sample)
                        samples.append(sample)
                        cur_samples += 1

                        if save:
                            self._save_sample(sample, self.save_dir, cur_samples)


class SeekNetC(SeekNetB):
    def __init__(self):
        super(SeekNetC, self).__init__()
        self.save_dir = Path.cwd() / "runs" / "identify" / "samples" / "seek_c_2"

        self.MARGIN = np.arange(0.15, 0.66, 0.05)
        self.SCALE = np.arange(32, 97, 8)
        self.EMBEDDING = np.arange(64, 1025, 64)

    def sample_design(self, num_samples: int = 100, save: bool = False):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        cur_samples = 0
        samples = []
        samples_per_range = num_samples // len(self.flops)

        for idx, target_flops in enumerate(self.flops):
            target_min = target_flops[0]
            target_max = target_flops[1]

            while cur_samples < samples_per_range * (idx + 1):
                raw_sample = self._generate_sample()
                sample = self._convert(raw_sample=raw_sample)

                if sample is not None and self._valid_sample(sample=sample):
                    flops = self._get_flops(sample)

                    if target_min <= flops <= target_max:
                        sample.update(raw_sample)
                        sample["flops"] = flops

                        sample = self._convert_np_to_native(sample)
                        samples.append(sample)
                        cur_samples += 1

                        if save:
                            self._save_sample(sample, self.save_dir, cur_samples)

    def _generate_sample(self) -> Dict:
        total_blocks = np.random.choice(self.TOTAL_BLOCKS)
        base_width = np.random.choice(self.BASE_WIDTH)
        width_increment = np.random.choice(self.WIDTH_INCREMENT)
        multip = np.random.choice(self.MULTIP)
        bias = np.random.choice(self.BIAS)
        smooth = np.random.choice(self.SMOOTH)

        bottleneck_ratio = np.random.choice(self.BOTTLENECK_RATIO)
        group_width = np.random.choice(self.GROUP_WIDTH)
        se_ratio = np.random.choice(self.SE_RATIO)

        margin = np.random.choice(self.MARGIN)
        scale = np.random.choice(self.SCALE)
        embedding = np.random.choice(self.EMBEDDING)

        return {
            "total_blocks": total_blocks,
            "base_width": base_width,
            "width_increment": width_increment,
            "multip": multip,
            "bias": bias,
            "smooth": smooth,
            "bottleneck_ratio": bottleneck_ratio,
            "group_width": group_width,
            "se_ratio": se_ratio,
            "m": margin,
            "s": scale,
            "embed_dim": embedding,
        }

    def _save_sample(self, sample: Dict, save_dir: Path, sample_id: int):
        file_path = save_dir / f"sample_{sample_id}.yaml"
        with open(file_path, "a") as f:
            yaml.dump(sample, f, default_flow_style=False, allow_unicode=True)
        print(f"Samples saved to {file_path}")


def optimize_a(type: str):
    base_dir = Path.cwd()
    data = base_dir / "cfg" / "datasets" / "fish-identify-n.yaml"
    samples_dir = base_dir / "runs" / "identify" / "samples" / "sample_a_no"
    s_dir = samples_dir / type

    sorted_path = sorted(s_dir.iterdir(), key=lambda x: int(x.stem.split("_")[1]))

    for path in sorted_path:
        with open(path) as f:
            config = yaml.safe_load(f)
            model = SHOUModel(type=path.parent.stem, config=config)

            new_dir = samples_dir / "train" / path.parent.stem / "train"
            index = 1
            while True:
                if not new_dir.exists():
                    break
                new_dir = new_dir.parent / f"train{index}"
                index += 1

            model._change_dir(new_dir)

            model.train(
                data=data,
                epochs=10,
                device=torch.device("cuda:0"),
                batch=128,
                lr0=1e-2,
                lrf=1e-2,
            )
            del model.model
            torch.cuda.empty_cache()


def optimize_b(ratio: int = 1):
    base_dir = Path.cwd()
    data = base_dir / "cfg" / "datasets" / "fish-identify-n.yaml"

    samples_dir = (
        base_dir / "runs" / "identify" / "samples" / "sample_b" / f"bottleneck_{ratio}"
    )
    sorted_dir = sorted(samples_dir.iterdir(), key=lambda x: int(x.stem.split("_")[1]))

    for path in sorted_dir:
        # if samples_dir.stem == "bottleneck_1" and int(path.stem.split("_")[1]) <= 97:
        #     continue

        with open(path) as f:
            config = yaml.safe_load(f)
            model = SHOUModel(type="CD", config=config)

            new_dir = samples_dir.parent / "train" / path.parent.stem / "train"
            index = 1
            while True:
                if not new_dir.exists():
                    break
                new_dir = new_dir.parent / f"train{index}"
                index += 1

            model._change_dir(new_dir)

            model.train(
                data=data,
                epochs=10,
                device=torch.device("cuda:1"),
                batch=128,
                lr0=1e-2,
                lrf=1e-2,
            )
            del model.model
            torch.cuda.empty_cache()


def optimize_c(group_width: int):
    base_dir = Path.cwd()
    data = base_dir / "cfg" / "datasets" / "fish-identify-n.yaml"

    samples_dir = (
        base_dir
        / "runs"
        / "identify"
        / "samples"
        / "sample_c"
        / f"groupw_{group_width}"
    )
    sorted_dir = sorted(samples_dir.iterdir(), key=lambda x: int(x.stem.split("_")[1]))

    for path in sorted_dir:
        with open(path) as f:
            config = yaml.safe_load(f)
            model = SHOUModel(type="CD", config=config)

            new_dir = samples_dir.parent / "train" / path.parent.stem / "train"
            index = 1
            while True:
                if not new_dir.exists():
                    break
                new_dir = new_dir.parent / f"train{index}"
                index += 1

            model._change_dir(new_dir)

            model.train(
                data=data,
                epochs=10,
                device=torch.device("cuda:1"),
                batch=128,
                lr0=1e-2,
                lrf=1e-2,
            )
            del model.model
            torch.cuda.empty_cache()


def optimize_d(se_ratio: int):
    base_dir = Path.cwd()
    data = base_dir / "cfg" / "datasets" / "fish-identify-n.yaml"

    samples_dir = (
        base_dir / "runs" / "identify" / "samples" / "sample_d" / f"se_{se_ratio}"
    )
    sorted_dir = sorted(samples_dir.iterdir(), key=lambda x: int(x.stem.split("_")[1]))

    for path in sorted_dir:
        with open(path) as f:
            config = yaml.safe_load(f)
            model = SHOUModel(type="CD", config=config)

            new_dir = samples_dir.parent / path.parent.stem / "train"
            index = 1
            while True:
                if not new_dir.exists():
                    break
                new_dir = new_dir.parent / f"train{index}"
                index += 1

            model._change_dir(new_dir)

            model.train(
                data=data,
                epochs=10,
                device=torch.device("cuda:1"),
                batch=128,
                lr0=1e-2,
                lrf=1e-2,
            )
            del model.model
            torch.cuda.empty_cache()


def process_path(path: Path, data: Path, cuda_id: int):
    with open(path) as f:
        config = yaml.safe_load(f)
        model = SHOUModel(type="CD", config=config)

        new_dir = path.parent / path.stem
        model._change_dir(new_dir)

        new_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, new_dir / path.name)

        model.train(
            data=data,
            epochs=25,
            device=torch.device(f"cuda:{cuda_id}"),
            workers=4,
            batch=12864,
            lr0=1e-2,
            lrf=1e-2,
        )

        del model.model
        torch.cuda.empty_cache()


def seek(samples_dir: Path, mode: int = 1):
    base_dir = Path.cwd()
    data = base_dir / "cfg" / "datasets" / "fish-identify-n.yaml"
    samples_dir = [s for s in samples_dir.iterdir() if not s.is_dir()]
    sorted_dir = sorted(samples_dir, key=lambda x: int(Path(x).stem.split("_")[1]))

    # mid = len(sorted_dir) // 2
    # tasks = sorted_dir[:mid] if mode == 0 else sorted_dir[mid:]
    tasks = sorted_dir[329]
    process_path(tasks, data, mode)

    # for file in tasks:
    #     process_path(file, data, mode)


def _a():
    save_path = ["X", "CX", "CX2", "CD"]
    for path in save_path:
        # samplenet = SampleNetA()
        # samplenet.sample_design(num_samples=200, type=path, save=True)
        optimize_a(path)


def _b():
    bottleneck_ratios = [1, 2, 4]
    for b in bottleneck_ratios:
        # samplenet = SampleNetB()
        # samplenet.sample_design(num_samples=200, bottleneck_ratio=b, save=True)

        optimize_b(ratio=b)


def _c():
    group_width = [1, 2, 4, 8, 16, 32]
    for c in group_width:
        # samplenet = SampleNetC()
        # samplenet.sample_design(num_samples=100, group_width=c, save=True)

        optimize_c(group_width=c)


def _d():
    se_ratio = [4, 8, 12, 16, 20, 24]
    for s in se_ratio:
        # samplenet = SampleNetD()
        # samplenet.sample_design(num_samples=100, se_ratio=s, save=True)

        optimize_d(se_ratio=s)


def _s():
    # seeknet = SeekNetA()
    # seeknet = SeekNetB()
    # seeknet.sample_design(num_samples=3000, save=True)
    seek()


def _s_b():
    # seeknet = SeekNetB()
    # seeknet.sample_design(num_samples=1500, save=True)

    samples_dir = Path.cwd() / "runs" / "identify" / "samples" / "seek_b_3"
    seek(samples_dir=samples_dir, mode=0)


def _s_c():
    # seeknet = SeekNetC()
    # seeknet.sample_design(num_samples=500, save=True)

    samples_dir = Path.cwd() / "runs" / "identify" / "samples" / "seek_c_2"
    seek(samples_dir=samples_dir, mode=1)


if __name__ == "__main__":
    # _a()
    # _b()
    _c()
    # _d()
    # _s()
    # _s_b()
    # _s_c()
