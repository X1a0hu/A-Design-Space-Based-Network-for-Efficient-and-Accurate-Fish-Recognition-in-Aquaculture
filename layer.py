import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Union, Tuple

sys.path.append(str(Path(__file__).parent))


def Conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    bias=False,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class SEBlock(nn.Module):

    def __init__(self, in_channels, ratio=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // ratio,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels // ratio,
                out_channels=in_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(x)
        return x * out


class BottleneckX(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_ratio,
        group_width: int,
        stride: int,
        se_ratio: int,
    ):
        super(BottleneckX, self).__init__()
        # Compute the number of groups
        hidden_channels = out_channels // bottleneck_ratio
        groups = max(hidden_channels // group_width, 1)

        self.silu = nn.SiLU(inplace=True)

        self.block_1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.block_2 = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=stride,
            groups=groups,
            padding=1,
        )
        self.se = SEBlock(in_channels=hidden_channels, ratio=se_ratio)
        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        out = self.block_1(x)
        out = self.block_2(out)
        out = self.se(out)
        out = self.block_3(out)

        x = self.shortcut(x)

        return self.silu(x + out)


class BottleneckD(nn.Module):

    def __init__(self, in_channels, out_channels, groups: int, ratio: int):
        """
        Bottleneck block with optional shortcut connection.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            groups (int): Number of groups for cov.
        """
        super(BottleneckD, self).__init__()

        self.silu = nn.SiLU(inplace=True)

        self.block_1 = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
        )
        self.se = SEBlock(in_channels=in_channels, ratio=ratio)
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block_1(x)
        out = self.se(out)
        out = self.block_2(out)

        return self.silu(x + out)


class CSPBlockX(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_ratio: int,
        group_width: int,
        num_blocks: int,
        se_ratio: int,
    ):
        super(CSPBlockX, self).__init__()
        hidden_channels = out_channels // bottleneck_ratio

        self.block_1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
        )
        # The hidden_channels will be recalculate in BottleneckX
        self.bottlenecks = nn.Sequential(
            *[
                BottleneckX(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bottleneck_ratio=1,
                    group_width=group_width,
                    stride=1,
                    se_ratio=se_ratio,
                )
                for _ in range(num_blocks)
            ]
        )

        self.block_2 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
        )

        self.block_3 = Conv(
            in_channels=2 * hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor):
        out_1 = self.block_1(x)
        out_1 = self.bottlenecks(out_1)
        out_2 = self.block_2(x)
        out_3 = torch.cat([out_1, out_2], dim=1)
        out_3 = self.block_3(out_3)

        return out_3


class CSPBlockX2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_ratio: int,
        group_width: int,
        num_blocks: int,
        se_ratio: int,
    ):
        super().__init__()
        hidden_channels = out_channels // bottleneck_ratio

        self.block_1 = Conv(
            in_channels=in_channels,
            out_channels=2 * hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bottlenecks = nn.ModuleList(
            BottleneckX(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                bottleneck_ratio=1,
                group_width=group_width,
                stride=1,
                se_ratio=se_ratio,
            )
            for _ in range(num_blocks)
        )
        self.block_3 = Conv(
            in_channels=(2 + num_blocks) * hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor):
        y = list(self.block_1(x).chunk(2, 1))
        y.extend(b(y[-1]) for b in self.bottlenecks)
        return self.block_3(torch.cat(y, 1))


class CSPBlockD(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_ratio: int,
        group_width: int,
        num_blocks: int,
        se_ratio: int,
    ):
        super().__init__()
        hidden_channels = out_channels // bottleneck_ratio
        groups = max(hidden_channels // group_width, 1)

        self.block_1 = Conv(
            in_channels=in_channels,
            out_channels=2 * hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bottlenecks = nn.ModuleList(
            BottleneckD(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                groups=groups,
                ratio=se_ratio,
            )
            for _ in range(num_blocks)
        )
        self.block_3 = Conv(
            in_channels=(2 + num_blocks) * hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor):
        y = list(self.block_1(x).chunk(2, 1))
        y.extend(b(y[-1]) for b in self.bottlenecks)
        return self.block_3(torch.cat(y, 1))
