import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import List

sys.path.append(str(Path(__file__).parent))

import layer


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        self.stem = layer.Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        return self.stem(x)


class StageX(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks: int,
        bottleneck_ratio: int,
        group_width: int,
        se_ratio: int,
    ):
        super(StageX, self).__init__()
        self.block_1 = layer.BottleneckX(
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck_ratio=bottleneck_ratio,
            group_width=group_width,
            stride=2,
            se_ratio=se_ratio,
        )

        self.bottlenecks = (
            nn.Sequential(
                *[
                    layer.BottleneckX(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        bottleneck_ratio=bottleneck_ratio,
                        group_width=group_width,
                        stride=1,
                        se_ratio=se_ratio,
                    )
                    for _ in range(num_blocks - 1)
                ]
            )
            if num_blocks - 1 > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.bottlenecks(x)
        return x


class StageCX(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks: int,
        bottleneck_ratio: int,
        group_width: int,
        se_ratio: int,
    ):
        super(StageCX, self).__init__()

        self.block_1 = layer.BottleneckX(
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck_ratio=bottleneck_ratio,
            group_width=group_width,
            stride=2,
            se_ratio=se_ratio,
        )
        self.block_2 = (
            layer.CSPBlockX(
                in_channels=out_channels,
                out_channels=out_channels,
                bottleneck_ratio=bottleneck_ratio,
                group_width=group_width,
                num_blocks=num_blocks - 1,
                se_ratio=se_ratio,
            )
            if num_blocks - 1 > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)

        return x


class StageCX2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks: int,
        bottleneck_ratio: int,
        group_width: int,
        se_ratio: int,
    ):
        super(StageCX2, self).__init__()
        self.block_1 = layer.BottleneckX(
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck_ratio=bottleneck_ratio,
            group_width=group_width,
            stride=2,
            se_ratio=se_ratio,
        )
        self.block_2 = (
            layer.CSPBlockX2(
                in_channels=out_channels,
                out_channels=out_channels,
                bottleneck_ratio=bottleneck_ratio,
                group_width=group_width,
                num_blocks=num_blocks - 1,
                se_ratio=se_ratio,
            )
            if num_blocks - 1 > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)

        return x


class StageCD(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks: int,
        bottleneck_ratio: int,
        group_width: int,
        se_ratio: int,
    ):
        super(StageCD, self).__init__()
        self.block_1 = layer.Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            groups=1,
            padding=1,
        )
        self.block_2 = (
            layer.CSPBlockD(
                in_channels=out_channels,
                out_channels=out_channels,
                bottleneck_ratio=bottleneck_ratio,
                group_width=group_width,
                num_blocks=num_blocks - 1,
                se_ratio=se_ratio,
            )
            if num_blocks - 1 > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        return x


class Body(nn.Module):

    def __init__(
        self,
        stage_type: str,
        stem_out: int,
        out_channels: List,
        num_blocks: List,
        bottleneck_ratios: List,
        group_widths: List,
        se_ratios: List,
    ):

        super(Body, self).__init__()
        self.stages = nn.ModuleList()

        stage_params = zip(
            out_channels,
            num_blocks,
            bottleneck_ratios,
            group_widths,
            se_ratios,
        )

        prev_channel = stem_out
        for (
            cur_channel,
            num_block,
            bottleneck_ratio,
            group_width,
            se_reatio,
        ) in stage_params:
            self.stages.append(
                self._create(
                    stage_type=stage_type,
                    in_channels=prev_channel,
                    out_channels=cur_channel,
                    num_blocks=num_block,
                    bottleneck_ratio=bottleneck_ratio,
                    group_width=group_width,
                    se_ratio=se_reatio,
                )
            )

            prev_channel = cur_channel

    def _create(
        self,
        stage_type: str,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        bottleneck_ratio: int,
        group_width: int,
        se_ratio: int,
    ):
        if stage_type == "X":
            return StageX(
                in_channels=in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                bottleneck_ratio=bottleneck_ratio,
                group_width=group_width,
                se_ratio=se_ratio,
            )
        elif stage_type == "CX":
            return StageCX(
                in_channels=in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                bottleneck_ratio=bottleneck_ratio,
                group_width=group_width,
                se_ratio=se_ratio,
            )
        elif stage_type == "CX2":
            return StageCX2(
                in_channels=in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                bottleneck_ratio=bottleneck_ratio,
                group_width=group_width,
                se_ratio=se_ratio,
            )
        elif stage_type == "CD":
            return StageCD(
                in_channels=in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                bottleneck_ratio=bottleneck_ratio,
                group_width=group_width,
                se_ratio=se_ratio,
            )
        else:
            raise ValueError(f"Unknown stage type: {stage_type}")

    def forward(self, x: torch.Tensor):
        for stage in self.stages:
            x = stage(x)
        return x


class Head(nn.Module):

    def __init__(self, in_channels, embed_dim):
        super(Head, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, embed_dim, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SeekNet(nn.Module):

    def __init__(
        self,
        stage_type: str,
        block_widths: List,
        num_blocks: List,
        bottleneck_ratios: List,
        group_widths: List,
        se_ratios: List,
        embed_dim,
    ):
        super(SeekNet, self).__init__()
        self.stem_w = 32

        self._validate_input_lengths(
            block_widths,
            num_blocks,
            bottleneck_ratios,
            group_widths,
            se_ratios,
        )

        self.stem = Stem(in_channels=3, out_channels=self.stem_w)
        self.body = Body(
            stage_type=stage_type,
            stem_out=self.stem_w,
            out_channels=block_widths,
            num_blocks=num_blocks,
            bottleneck_ratios=bottleneck_ratios,
            group_widths=group_widths,
            se_ratios=se_ratios,
        )
        self.head = Head(in_channels=block_widths[-1], embed_dim=embed_dim)

    def _validate_input_lengths(
        self,
        block_widths,
        num_blocks,
        bottleneck_ratios,
        group_widths,
        se_ratios,
    ):
        """Ensures that all lists have the same length."""

        assert (
            len(block_widths)
            == len(num_blocks)
            == len(bottleneck_ratios)
            == len(group_widths)
            == len(se_ratios)
        ), "All input lists must have the same length."

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)

        return x


if __name__ == "__main__":
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    block_widths = [24, 56, 152, 368]
    num_blocks = [1, 1, 4, 7]
    bottleneck_ratios = [1, 1, 1, 1]
    group_widths = [8, 8, 8, 8]
    se_ratios = [8, 8, 8, 8]
    embed_dim = 512

    "X, CX, CX2, CD"
    model = SeekNet(
        stage_type="X",
        block_widths=block_widths,
        num_blocks=num_blocks,
        bottleneck_ratios=bottleneck_ratios,
        group_widths=group_widths,
        se_ratios=se_ratios,
        embed_dim=embed_dim,
    )

    # for name, module in model.named_children():
    #     print(name, "->", module)

    model.to(device="cuda")
    summary(model=model, input_size=(3, 224, 224))

    flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).to("cuda"))
    print(f"Total GFLOPs: {flops.total() / 1e9:.4f}")
