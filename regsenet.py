import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Stage(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks: int,
        bottleneck_ratio: int,
        group_width: int,
        se_ratio: int,
    ):
        super(Stage, self).__init__()
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            channel = in_channels if i == 0 else out_channels

            stride = 2 if i == 0 else 1

            self.blocks.append(
                layer.XBlock(
                    in_channels=channel,
                    out_channels=out_channels,
                    bottleneck_ratio=bottleneck_ratio,
                    stride=stride,
                    group_width=group_width,
                    se_ratio=se_ratio,
                )
            )

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x


class Body(nn.Module):

    def __init__(
        self,
        stem_out: int,
        out_channels: List,
        num_blocks: List,
        bottleneck_ratios: List,
        group_widths: int,
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
            channel,
            num_block,
            bottleneck_ratio,
            group_width,
            se_ratio,
        ) in stage_params:
            self.stages.append(
                Stage(
                    in_channels=prev_channel,
                    out_channels=channel,
                    num_blocks=num_block,
                    bottleneck_ratio=bottleneck_ratio,
                    group_width=group_width,
                    se_ratio=se_ratio,
                )
            )
            prev_channel = channel

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


class RegSENet(nn.Module):

    def __init__(
        self,
        out_channels: List,
        num_blocks: List,
        bottleneck_ratios: List,
        group_widths: List,
        se_ratios: List,
        embed_dim,
    ):

        self.stem_w = 32
        super(RegSENet, self).__init__()

        self.stem = Stem(in_channels=3, out_channels=self.stem_w)
        self.body = Body(
            stem_out=self.stem_w,
            out_channels=out_channels,
            num_blocks=num_blocks,
            bottleneck_ratios=bottleneck_ratios,
            group_widths=group_widths,
            se_ratios=se_ratios,
        )

        self.head = Head(in_channels=out_channels[-1], embed_dim=embed_dim)

    def _validate_input_lengths(
        self,
        out_channels,
        num_blocks,
        se_ratios,
    ):
        """Ensures that all lists have the same length."""
        assert (
            len(out_channels) == len(num_blocks) == len(se_ratios)
        ), "All input lists must have the same length."

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)

        return x


if __name__ == "__main__":
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    block_widths = [16, 88, 8, 56]
    num_blocks = [4, 5, 6, 4]
    bottleneck_ratios = [2, 1, 2, 1]
    group_widths = [16, 32, 1, 32]
    se_ratios = [8, 8, 4, 8]
    embed_dim = 512

    model = RegSENet(
        out_channels=block_widths,
        num_blocks=num_blocks,
        bottleneck_ratios=bottleneck_ratios,
        group_widths=group_widths,
        se_ratios=se_ratios,
        embed_dim=embed_dim,
    )
    model.to(device="cuda")
    summary(model=model, input_size=(3, 224, 224))

    flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).to("cuda"))
    print(f"Total GFLOPs: {flops.total() / 1e9:.4f}")
