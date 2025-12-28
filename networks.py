import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))


def Conv1(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    )


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = Conv1(
            in_channel=3, out_channel=32, kernel_size=3, stride=2, padding=0
        )
        self.conv2 = Conv1(
            in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=0
        )
        self.conv3 = Conv1(
            in_channel=32, out_channel=64, kernel_size=3, stride=1, padding=1
        )
        self.branch1_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch1_2 = Conv1(
            in_channel=64, out_channel=96, kernel_size=3, stride=2, padding=0
        )
        self.branch2_1_1 = Conv1(
            in_channel=160, out_channel=64, kernel_size=1, stride=1, padding=0
        )
        self.branch2_1_2 = Conv1(
            in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0
        )
        self.branch2_2_1 = Conv1(
            in_channel=160, out_channel=64, kernel_size=1, stride=1, padding=0
        )
        self.branch2_2_2 = Conv1(
            in_channel=64, out_channel=64, kernel_size=(7, 1), stride=1, padding=(3, 0)
        )
        self.branch2_2_3 = Conv1(
            in_channel=64, out_channel=64, kernel_size=(1, 7), stride=1, padding=(0, 3)
        )
        self.branch2_2_4 = Conv1(
            in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0
        )
        self.branch3_1 = Conv1(
            in_channel=192, out_channel=192, kernel_size=3, stride=2, padding=0
        )
        self.branch3_2 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4_1 = self.branch1_1(out3)
        out4_2 = self.branch1_2(out3)
        out4 = torch.cat((out4_1, out4_2), dim=1)
        out5_1 = self.branch2_1_2(self.branch2_1_1(out4))
        out5_2 = self.branch2_2_4(
            self.branch2_2_3(self.branch2_2_2(self.branch2_2_1(out4)))
        )
        out5 = torch.cat((out5_1, out5_2), dim=1)
        out6_1 = self.branch3_1(out5)
        out6_2 = self.branch3_2(out5)
        out = torch.cat((out6_1, out6_2), dim=1)
        return out


class InceptionCBAMResNetA(nn.Module):
    def __init__(self, in_channel, scale=0.1):
        super(InceptionCBAMResNetA, self).__init__()
        self.out_channel = 384
        self.scale = scale

        self.branch1 = Conv1(
            in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0
        )
        self.branch2_1 = Conv1(
            in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0
        )
        self.branch2_2 = Conv1(
            in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=1
        )
        self.branch3_1 = Conv1(
            in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0
        )
        self.branch3_2 = Conv1(
            in_channel=32, out_channel=48, kernel_size=3, stride=1, padding=1
        )
        self.branch3_3 = Conv1(
            in_channel=48, out_channel=64, kernel_size=3, stride=1, padding=1
        )
        self.linear = Conv1(
            in_channel=128, out_channel=384, kernel_size=1, stride=1, padding=0
        )
        self.cbam = CBAMLayer(in_channel=self.out_channel)
        # self.se = SE(inchannel=self.out_channel)

        self.shortcut = nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=self.out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output1 = self.branch1(x)
        output2 = self.branch2_2(self.branch2_1(x))
        output3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))
        out = torch.cat((output1, output2, output3), dim=1)
        out = self.linear(out)
        # out = self.se(out)
        out = self.cbam(out)
        out = self.shortcut(x) + self.scale * out
        out = F.relu(out)
        return out


class InceptionResNetB(nn.Module):
    def __init__(self, in_channel, scale=0.1):
        super(InceptionResNetB, self).__init__()
        self.out_channel = 1152
        self.scale = scale
        self.branch1 = Conv1(
            in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0
        )
        self.branch2_1 = Conv1(
            in_channel=in_channel, out_channel=128, kernel_size=1, stride=1, padding=0
        )
        self.branch2_2 = Conv1(
            in_channel=128,
            out_channel=160,
            kernel_size=(1, 7),
            stride=1,
            padding=(0, 3),
        )
        self.branch2_3 = Conv1(
            in_channel=160,
            out_channel=192,
            kernel_size=(7, 1),
            stride=1,
            padding=(3, 0),
        )
        self.linear = Conv1(
            in_channel=384, out_channel=1152, kernel_size=1, stride=1, padding=0
        )
        # self.se = SE(inchannel=self.out_channel)
        # self.cbam = CBAMLayer(in_channel=self.out_channel)

        self.shortcut = nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=self.out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output1 = self.branch1(x)
        output2 = self.branch2_3(self.branch2_2(self.branch2_1(x)))
        out = torch.cat((output1, output2), dim=1)
        out = self.linear(out)
        # out = self.se(out)
        # out = self.cbam(out)
        out = self.shortcut(x) + self.scale * out
        out = F.relu(out)
        return out


class InceptionSEResNetC(nn.Module):
    def __init__(self, in_channel, scale=0.1):
        self.out_channel = 2144
        self.scale = scale
        super(InceptionSEResNetC, self).__init__()
        self.branch1 = Conv1(
            in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0
        )
        self.branch2_1 = Conv1(
            in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0
        )
        self.branch2_2 = Conv1(
            in_channel=192,
            out_channel=224,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )
        self.branch2_3 = Conv1(
            in_channel=224,
            out_channel=256,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )
        self.linear = Conv1(
            in_channel=448, out_channel=2144, kernel_size=1, stride=1, padding=0
        )
        self.se = SELayer(inchannel=self.out_channel)
        # self.cbam = CBAMLayer(in_channel=self.out_channel)

        self.shortcut = nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=self.out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output1 = self.branch1(x)
        output2 = self.branch2_3(self.branch2_2(self.branch2_1(x)))
        out = torch.cat((output1, output2), dim=1)
        out = self.linear(out)
        out = self.se(out)
        # out = self.cbam(out)
        out = self.shortcut(x) + out * self.scale
        out = F.relu(out)
        return out


class ReductionA(nn.Module):
    def __init__(self, in_channel):
        super(ReductionA, self).__init__()
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch2 = Conv1(
            in_channel=in_channel, out_channel=384, kernel_size=3, stride=2, padding=0
        )
        self.branch3_1 = Conv1(
            in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
        )
        self.branch3_2 = Conv1(
            in_channel=256, out_channel=256, kernel_size=3, stride=1, padding=1
        )
        self.branch3_3 = Conv1(
            in_channel=256, out_channel=384, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))
        return torch.cat((out1, out2, out3), dim=1)


class ReductionB(nn.Module):
    def __init__(self, in_channel):
        super(ReductionB, self).__init__()
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch2_1 = Conv1(
            in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
        )
        self.branch2_2 = Conv1(
            in_channel=256, out_channel=384, kernel_size=3, stride=2, padding=0
        )
        self.branch3_1 = Conv1(
            in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
        )
        self.branch3_2 = Conv1(
            in_channel=256, out_channel=288, kernel_size=3, stride=2, padding=0
        )
        self.branch4_1 = Conv1(
            in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
        )
        self.branch4_2 = Conv1(
            in_channel=256, out_channel=288, kernel_size=3, stride=1, padding=1
        )
        self.branch4_3 = Conv1(
            in_channel=288, out_channel=320, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2_2(self.branch2_1(x))
        out3 = self.branch3_2(self.branch3_1(x))
        out4 = self.branch4_3(self.branch4_2(self.branch4_1(x)))
        return torch.cat((out1, out2, out3, out4), dim=1)


class SELayer(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=True),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CBAMLayer(nn.Module):
    def __init__(self, in_channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, in_channel, 1, bias=True),
        )
        # spatial attention
        self.conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class SHOUNet(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, dropout_prob: int):
        super(SHOUNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        blocks = nn.ModuleList()
        blocks.append(Stem())

        # 5 InceptionCBAMResNetA
        for _ in range(5):
            blocks.append(InceptionCBAMResNetA(384))
        blocks.append(ReductionA(384))

        # 10 InceptionResNetB
        for _ in range(10):
            blocks.append(InceptionResNetB(1152))
        blocks.append(ReductionB(1152))

        # 5 InceptionSEResNetC
        for _ in range(5):
            blocks.append(InceptionSEResNetC(2144))

        self.map = nn.Sequential(*blocks)
        self.pool = nn.AvgPool2d(kernel_size=8)
        self.dropout = nn.Dropout(self.dropout_prob)

        # GradCAM
        self.target_layers = [
            blocks[-1],
        ]

        # Embedding head
        self.embedding_head = nn.Linear(2144, self.embedding_dim)

    def set_emb_head(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.embedding_head = nn.Linear(2144, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.map(x)
        x = self.pool(x)
        x = self.dropout(x)
        features = x.view(x.size(0), -1)

        embeddings = self.embedding_head(features)

        return embeddings

    def predict(self, x: torch.Tensor):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# import sys
# from pathlib import Path
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# sys.path.append(str(Path(__file__).parent))


# def Conv1(in_channel, out_channel, kernel_size, stride, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
#         nn.BatchNorm2d(out_channel),
#         nn.ReLU(),
#     )


# class Stem(nn.Module):
#     def __init__(self):
#         super(Stem, self).__init__()
#         self.conv1 = Conv1(
#             in_channel=3, out_channel=32, kernel_size=3, stride=2, padding=0
#         )
#         self.conv2 = Conv1(
#             in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=0
#         )
#         self.conv3 = Conv1(
#             in_channel=32, out_channel=64, kernel_size=3, stride=1, padding=1
#         )
#         self.branch1_1 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.branch1_2 = Conv1(
#             in_channel=64, out_channel=96, kernel_size=3, stride=2, padding=0
#         )
#         self.branch2_1_1 = Conv1(
#             in_channel=160, out_channel=64, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_1_2 = Conv1(
#             in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0
#         )
#         self.branch2_2_1 = Conv1(
#             in_channel=160, out_channel=64, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_2_2 = Conv1(
#             in_channel=64, out_channel=64, kernel_size=(7, 1), stride=1, padding=(3, 0)
#         )
#         self.branch2_2_3 = Conv1(
#             in_channel=64, out_channel=64, kernel_size=(1, 7), stride=1, padding=(0, 3)
#         )
#         self.branch2_2_4 = Conv1(
#             in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0
#         )
#         self.branch3_1 = Conv1(
#             in_channel=192, out_channel=192, kernel_size=3, stride=2, padding=0
#         )
#         self.branch3_2 = nn.MaxPool2d(kernel_size=3, stride=2)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out1 = self.conv1(x)
#         out2 = self.conv2(out1)
#         out3 = self.conv3(out2)
#         out4_1 = self.branch1_1(out3)
#         out4_2 = self.branch1_2(out3)
#         out4 = torch.cat((out4_1, out4_2), dim=1)
#         out5_1 = self.branch2_1_2(self.branch2_1_1(out4))
#         out5_2 = self.branch2_2_4(
#             self.branch2_2_3(self.branch2_2_2(self.branch2_2_1(out4)))
#         )
#         out5 = torch.cat((out5_1, out5_2), dim=1)
#         out6_1 = self.branch3_1(out5)
#         out6_2 = self.branch3_2(out5)
#         out = torch.cat((out6_1, out6_2), dim=1)
#         return out


# class InceptionCBAMResNetA(nn.Module):
#     def __init__(self, in_channel, scale=0.17):
#         super(InceptionCBAMResNetA, self).__init__()
#         self.out_channel = 384
#         self.scale = scale

#         self.branch1 = Conv1(
#             in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_1 = Conv1(
#             in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_2 = Conv1(
#             in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=1
#         )
#         self.branch3_1 = Conv1(
#             in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0
#         )
#         self.branch3_2 = Conv1(
#             in_channel=32, out_channel=48, kernel_size=3, stride=1, padding=1
#         )
#         self.branch3_3 = Conv1(
#             in_channel=48, out_channel=64, kernel_size=3, stride=1, padding=1
#         )
#         self.linear = Conv1(
#             in_channel=128, out_channel=384, kernel_size=1, stride=1, padding=0
#         )
#         self.cbam = CBAMLayer(in_channel=self.out_channel)
#         # self.se = SE(inchannel=self.out_channel)

#         self.shortcut = nn.Sequential()
#         if in_channel != self.out_channel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=in_channel,
#                     out_channels=self.out_channel,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0,
#                 ),
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         output1 = self.branch1(x)
#         output2 = self.branch2_2(self.branch2_1(x))
#         output3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))
#         out = torch.cat((output1, output2, output3), dim=1)
#         out = self.linear(out)
#         # out = self.se(out)
#         out = self.cbam(out)
#         out = self.shortcut(x) + self.scale * out
#         out = F.relu(out)
#         return out


# class InceptionResNetB(nn.Module):
#     def __init__(self, in_channel, scale=0.1):
#         super(InceptionResNetB, self).__init__()
#         self.out_channel = 1152
#         self.scale = scale
#         self.branch1 = Conv1(
#             in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_1 = Conv1(
#             in_channel=in_channel, out_channel=128, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_2 = Conv1(
#             in_channel=128,
#             out_channel=160,
#             kernel_size=(1, 7),
#             stride=1,
#             padding=(0, 3),
#         )
#         self.branch2_3 = Conv1(
#             in_channel=160,
#             out_channel=192,
#             kernel_size=(7, 1),
#             stride=1,
#             padding=(3, 0),
#         )
#         self.linear = Conv1(
#             in_channel=384, out_channel=1152, kernel_size=1, stride=1, padding=0
#         )
#         # self.se = SE(inchannel=self.out_channel)
#         # self.cbam = CBAMLayer(in_channel=self.out_channel)

#         self.shortcut = nn.Sequential()
#         if in_channel != self.out_channel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=in_channel,
#                     out_channels=self.out_channel,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0,
#                 )
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         output1 = self.branch1(x)
#         output2 = self.branch2_3(self.branch2_2(self.branch2_1(x)))
#         out = torch.cat((output1, output2), dim=1)
#         out = self.linear(out)
#         # out = self.se(out)
#         # out = self.cbam(out)
#         out = self.shortcut(x) + self.scale * out
#         out = F.relu(out)
#         return out


# class InceptionSEResNetC(nn.Module):
#     def __init__(self, in_channel, scale=0.2, activate=False):
#         self.out_channel = 2144
#         self.scale = scale
#         self.activate = activate

#         super(InceptionSEResNetC, self).__init__()
#         self.branch1 = Conv1(
#             in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_1 = Conv1(
#             in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_2 = Conv1(
#             in_channel=192,
#             out_channel=224,
#             kernel_size=(1, 3),
#             stride=1,
#             padding=(0, 1),
#         )
#         self.branch2_3 = Conv1(
#             in_channel=224,
#             out_channel=256,
#             kernel_size=(3, 1),
#             stride=1,
#             padding=(1, 0),
#         )
#         self.linear = Conv1(
#             in_channel=448, out_channel=2144, kernel_size=1, stride=1, padding=0
#         )
#         self.se = SELayer(inchannel=self.out_channel)
#         # self.cbam = CBAMLayer(in_channel=self.out_channel)

#         self.shortcut = nn.Sequential()
#         if in_channel != self.out_channel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=in_channel,
#                     out_channels=self.out_channel,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0,
#                 )
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         output1 = self.branch1(x)
#         output2 = self.branch2_3(self.branch2_2(self.branch2_1(x)))
#         out = torch.cat((output1, output2), dim=1)
#         out = self.linear(out)
#         out = self.se(out)
#         # out = self.cbam(out)
#         if self.activate:
#             return F.relu(self.shortcut(x) + out * self.scale)
#         return self.shortcut(x) + out * self.scale


# class ReductionA(nn.Module):
#     def __init__(self, in_channel):
#         super(ReductionA, self).__init__()
#         self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
#         self.branch2 = Conv1(
#             in_channel=in_channel, out_channel=384, kernel_size=3, stride=2, padding=0
#         )
#         self.branch3_1 = Conv1(
#             in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
#         )
#         self.branch3_2 = Conv1(
#             in_channel=256, out_channel=256, kernel_size=3, stride=1, padding=1
#         )
#         self.branch3_3 = Conv1(
#             in_channel=256, out_channel=384, kernel_size=3, stride=2, padding=0
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out1 = self.branch1(x)
#         out2 = self.branch2(x)
#         out3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))
#         return torch.cat((out1, out2, out3), dim=1)


# class ReductionB(nn.Module):
#     def __init__(self, in_channel):
#         super(ReductionB, self).__init__()
#         self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
#         self.branch2_1 = Conv1(
#             in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_2 = Conv1(
#             in_channel=256, out_channel=384, kernel_size=3, stride=2, padding=0
#         )
#         self.branch3_1 = Conv1(
#             in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
#         )
#         self.branch3_2 = Conv1(
#             in_channel=256, out_channel=288, kernel_size=3, stride=2, padding=0
#         )
#         self.branch4_1 = Conv1(
#             in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
#         )
#         self.branch4_2 = Conv1(
#             in_channel=256, out_channel=288, kernel_size=3, stride=1, padding=1
#         )
#         self.branch4_3 = Conv1(
#             in_channel=288, out_channel=320, kernel_size=3, stride=2, padding=0
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out1 = self.branch1(x)
#         out2 = self.branch2_2(self.branch2_1(x))
#         out3 = self.branch3_2(self.branch3_1(x))
#         out4 = self.branch4_3(self.branch4_2(self.branch4_1(x)))
#         return torch.cat((out1, out2, out3, out4), dim=1)


# class SELayer(nn.Module):
#     def __init__(self, inchannel, ratio=16):
#         super(SELayer, self).__init__()
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Sequential(
#             nn.Linear(inchannel, inchannel // ratio, bias=False),
#             nn.ReLU(),
#             nn.Linear(inchannel // ratio, inchannel, bias=False),
#             nn.Sigmoid(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, c, _, _ = x.size()
#         y = self.gap(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y


# class CBAMLayer(nn.Module):
#     def __init__(self, in_channel, reduction=16, spatial_kernel=7):
#         super(CBAMLayer, self).__init__()

#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         # shared MLP
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channel // reduction, in_channel, 1, bias=False),
#         )
#         # spatial attention
#         self.conv = nn.Conv2d(
#             2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         max_out = self.mlp(self.max_pool(x))
#         avg_out = self.mlp(self.avg_pool(x))
#         channel_out = self.sigmoid(max_out + avg_out)
#         x = channel_out * x

#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
#         x = spatial_out * x
#         return x


# class SHOUNet(nn.Module):
#     def __init__(self, embedding_dim: int, num_classes: int, dropout_prob: int):
#         super(SHOUNet, self).__init__()

#         self.embedding_dim = embedding_dim
#         self.num_classes = num_classes
#         self.dropout_prob = dropout_prob

#         blocks = nn.ModuleList()
#         blocks.append(Stem())

#         # 10 InceptionCBAMResNetA
#         for _ in range(5):
#             blocks.append(InceptionCBAMResNetA(384))
#         blocks.append(ReductionA(384))

#         # 20 InceptionResNetB
#         for _ in range(10):
#             blocks.append(InceptionResNetB(1152))
#         blocks.append(ReductionB(1152))

#         # 10 InceptionSEResNetC
#         for _ in range(4):
#             blocks.append(InceptionSEResNetC(2144))
#         blocks.append(InceptionSEResNetC(2144, activate=True))

#         self.map = nn.Sequential(*blocks)
#         self.pool = nn.AvgPool2d(kernel_size=8)
#         self.dropout = nn.Dropout(self.dropout_prob)

#         # GradCAM
#         self.target_layers = [
#             blocks[-1],
#         ]

#         # Embedding head
#         self.embedding_head = nn.Linear(2144, self.embedding_dim)

#         # Classfication head
#         self.classification_head = nn.Linear(2144, self.num_classes)

#     def set_emb_head(self, embedding_dim: int):
#         self.embedding_dim = embedding_dim
#         self.embedding_head = nn.Linear(2144, self.embedding_dim)

#     def set_cls_head(self, num_classes: int):
#         self.num_classes = num_classes
#         self.classification_head = nn.Linear(2144, self.num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.map(x)
#         x = self.pool(x)
#         x = self.dropout(x)
#         features = x.view(x.size(0), -1)

#         embeddings = self.embedding_head(features)
#         logits = self.classification_head(features)

#         return embeddings, logits

#     def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         self.eval()
#         with torch.no_grad():
#             return self.forward(x)


# import sys
# from pathlib import Path
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# sys.path.append(str(Path(__file__).parent))


# def StanConv(in_channel, out_channel, kernel_size, stride, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
#         nn.BatchNorm2d(out_channel),
#         nn.ReLU(inplace=True),
#     )


# def DepwiseConv(in_channel, out_channel, kernel_size, stride, padding):
#     return nn.Sequential(
#         nn.Conv2d(
#             in_channel,
#             in_channel,
#             kernel_size,
#             stride,
#             padding,
#             groups=in_channel,
#             bias=False,
#         ),
#         nn.BatchNorm2d(in_channel),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(
#             in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False
#         ),
#         nn.BatchNorm2d(out_channel),
#         nn.ReLU(inplace=True),
#     )


# class Stem(nn.Module):
#     def __init__(self):
#         super(Stem, self).__init__()
#         self.conv1 = DepwiseConv(
#             in_channel=3, out_channel=32, kernel_size=3, stride=2, padding=0
#         )
#         self.conv2 = DepwiseConv(
#             in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=0
#         )
#         self.conv3 = DepwiseConv(
#             in_channel=32, out_channel=64, kernel_size=3, stride=1, padding=1
#         )

#         self.branch1_1 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.branch1_2 = DepwiseConv(
#             in_channel=64, out_channel=96, kernel_size=3, stride=2, padding=0
#         )

#         self.branch2_1_1 = StanConv(
#             in_channel=160, out_channel=64, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_1_2 = DepwiseConv(
#             in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0
#         )

#         self.branch2_2_1 = StanConv(
#             in_channel=160, out_channel=64, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_2_2 = DepwiseConv(
#             in_channel=64, out_channel=64, kernel_size=(7, 1), stride=1, padding=(3, 0)
#         )
#         self.branch2_2_3 = DepwiseConv(
#             in_channel=64, out_channel=64, kernel_size=(1, 7), stride=1, padding=(0, 3)
#         )
#         self.branch2_2_4 = DepwiseConv(
#             in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0
#         )

#         self.branch3_1 = DepwiseConv(
#             in_channel=192, out_channel=192, kernel_size=3, stride=2, padding=0
#         )
#         self.branch3_2 = nn.MaxPool2d(kernel_size=3, stride=2)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out1 = self.conv3(self.conv2(self.conv1(x)))
#         out2_1 = self.branch1_1(out1)
#         out2_2 = self.branch1_2(out1)
#         out3 = torch.cat((out2_1, out2_2), dim=1)
#         out4_1 = self.branch2_1_2(self.branch2_1_1(out3))
#         out4_2 = self.branch2_2_4(
#             self.branch2_2_3(self.branch2_2_2(self.branch2_2_1(out3)))
#         )
#         out5 = torch.cat((out4_1, out4_2), dim=1)
#         out6_1 = self.branch3_1(out5)
#         out6_2 = self.branch3_2(out5)
#         out = torch.cat((out6_1, out6_2), dim=1)
#         return out


# class InceptionNetA(nn.Module):
#     def __init__(self, in_channel, scale=0.17):
#         super(InceptionNetA, self).__init__()
#         self.out_channel = 384
#         self.scale = scale

#         self.branch1 = StanConv(
#             in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2 = nn.Sequential(
#             StanConv(
#                 in_channel=in_channel,
#                 out_channel=32,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#             ),
#             DepwiseConv(
#                 in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=1
#             ),
#         )
#         self.branch3 = nn.Sequential(
#             StanConv(
#                 in_channel=in_channel,
#                 out_channel=32,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#             ),
#             DepwiseConv(
#                 in_channel=32, out_channel=48, kernel_size=3, stride=1, padding=1
#             ),
#             DepwiseConv(
#                 in_channel=48, out_channel=64, kernel_size=3, stride=1, padding=1
#             ),
#         )
#         self.linear = StanConv(
#             in_channel=128, out_channel=384, kernel_size=1, stride=1, padding=0
#         )
#         self.cbam = CBAMLayer(in_channel=self.out_channel)
#         # self.se = SELayer(inchannel=self.out_channel)

#         self.shortcut = nn.Sequential()
#         if in_channel != self.out_channel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=in_channel,
#                     out_channels=self.out_channel,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0,
#                     bias=True,
#                 ),
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out1 = self.branch1(x)
#         out2 = self.branch2(x)
#         out3 = self.branch3(x)
#         out = torch.cat((out1, out2, out3), dim=1)
#         out = self.linear(out)
#         out = self.cbam(out)
#         # out = self.se(out)
#         out = self.shortcut(x) + self.scale * out
#         return F.relu(out)


# class InceptionB(nn.Module):
#     def __init__(self, in_channel, scale=0.1):
#         super(InceptionB, self).__init__()
#         self.out_channel = 1152
#         self.scale = scale

#         self.branch1 = StanConv(
#             in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_1 = StanConv(
#             in_channel=in_channel, out_channel=128, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_2 = DepwiseConv(
#             in_channel=128,
#             out_channel=160,
#             kernel_size=(1, 7),
#             stride=1,
#             padding=(0, 3),
#         )
#         self.branch2_3 = DepwiseConv(
#             in_channel=160,
#             out_channel=192,
#             kernel_size=(7, 1),
#             stride=1,
#             padding=(3, 0),
#         )
#         self.linear = StanConv(
#             in_channel=384, out_channel=1152, kernel_size=1, stride=1, padding=0
#         )
#         self.se = SELayer(inchannel=self.out_channel)
#         # self.cbam = CBAMLayer(in_channel=self.out_channel)

#         self.shortcut = nn.Sequential()
#         if in_channel != self.out_channel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=in_channel,
#                     out_channels=self.out_channel,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0,
#                     bias=True,
#                 ),
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         output1 = self.branch1(x)
#         output2 = self.branch2_3(self.branch2_2(self.branch2_1(x)))
#         out = torch.cat((output1, output2), dim=1)
#         out = self.linear(out)
#         out = self.se(out)
#         # out = self.cbam(out)
#         out = self.shortcut(x) + self.scale * out
#         return F.relu(out)


# class InceptionC(nn.Module):
#     def __init__(self, in_channel, scale=0.2):
#         self.out_channel = 2144
#         self.scale = scale

#         super(InceptionC, self).__init__()
#         self.branch1 = StanConv(
#             in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_1 = StanConv(
#             in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_2 = DepwiseConv(
#             in_channel=192,
#             out_channel=224,
#             kernel_size=(1, 3),
#             stride=1,
#             padding=(0, 1),
#         )
#         self.branch2_3 = DepwiseConv(
#             in_channel=224,
#             out_channel=256,
#             kernel_size=(3, 1),
#             stride=1,
#             padding=(1, 0),
#         )
#         self.linear = StanConv(
#             in_channel=448, out_channel=2144, kernel_size=1, stride=1, padding=0
#         )
#         self.se = SELayer(inchannel=self.out_channel)
#         # self.cbam = CBAMLayer(in_channel=self.out_channel)

#         self.shortcut = nn.Sequential()
#         if in_channel != self.out_channel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=in_channel,
#                     out_channels=self.out_channel,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0,
#                     bias=True,
#                 ),
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out1 = self.branch1(x)
#         out2 = self.branch2_3(self.branch2_2(self.branch2_1(x)))
#         out = torch.cat((out1, out2), dim=1)
#         out = self.linear(out)
#         out = self.se(out)
#         # out = self.cbam(out)
#         out = self.shortcut(x) + self.scale * out
#         return F.relu(out)


# class ReductionA(nn.Module):
#     def __init__(self, in_channel):
#         super(ReductionA, self).__init__()
#         self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
#         self.branch2 = DepwiseConv(
#             in_channel=in_channel, out_channel=384, kernel_size=3, stride=2, padding=0
#         )
#         self.branch3_1 = StanConv(
#             in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
#         )
#         self.branch3_2 = DepwiseConv(
#             in_channel=256, out_channel=256, kernel_size=3, stride=1, padding=1
#         )
#         self.branch3_3 = DepwiseConv(
#             in_channel=256, out_channel=384, kernel_size=3, stride=2, padding=0
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out1 = self.branch1(x)
#         out2 = self.branch2(x)
#         out3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))
#         return torch.cat((out1, out2, out3), dim=1)


# class ReductionB(nn.Module):
#     def __init__(self, in_channel):
#         super(ReductionB, self).__init__()
#         self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
#         self.branch2_1 = StanConv(
#             in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
#         )
#         self.branch2_2 = DepwiseConv(
#             in_channel=256, out_channel=384, kernel_size=3, stride=2, padding=0
#         )
#         self.branch3_1 = StanConv(
#             in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
#         )
#         self.branch3_2 = DepwiseConv(
#             in_channel=256, out_channel=288, kernel_size=3, stride=2, padding=0
#         )
#         self.branch4_1 = StanConv(
#             in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0
#         )
#         self.branch4_2 = DepwiseConv(
#             in_channel=256, out_channel=288, kernel_size=3, stride=1, padding=1
#         )
#         self.branch4_3 = DepwiseConv(
#             in_channel=288, out_channel=320, kernel_size=3, stride=2, padding=0
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out1 = self.branch1(x)
#         out2 = self.branch2_2(self.branch2_1(x))
#         out3 = self.branch3_2(self.branch3_1(x))
#         out4 = self.branch4_3(self.branch4_2(self.branch4_1(x)))
#         return torch.cat((out1, out2, out3, out4), dim=1)


# class SELayer(nn.Module):
#     def __init__(self, inchannel, ratio=16):
#         super(SELayer, self).__init__()
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Sequential(
#             nn.Linear(inchannel, inchannel // ratio, bias=False),
#             nn.ReLU(),
#             nn.Linear(inchannel // ratio, inchannel, bias=False),
#             nn.Sigmoid(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, c, _, _ = x.size()
#         y = self.gap(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y


# class CBAMLayer(nn.Module):
#     def __init__(self, in_channel, reduction=16, spatial_kernel=7):
#         super(CBAMLayer, self).__init__()

#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         # shared MLP
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channel // reduction, in_channel, 1, bias=False),
#         )
#         # spatial attention
#         self.conv = nn.Conv2d(
#             2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         max_out = self.mlp(self.max_pool(x))
#         avg_out = self.mlp(self.avg_pool(x))
#         channel_out = self.sigmoid(max_out + avg_out)
#         x = channel_out * x

#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
#         x = spatial_out * x
#         return x


# class SHOUNet(nn.Module):
#     def __init__(self, embedding_dim: int, num_classes: int, dropout_prob: int):
#         super(SHOUNet, self).__init__()

#         self.embedding_dim = embedding_dim
#         self.num_classes = num_classes
#         self.dropout_prob = dropout_prob

#         blocks = nn.ModuleList()
#         blocks.append(Stem())

#         # 5 InceptionA
#         for _ in range(5):
#             blocks.append(InceptionNetA(384))
#         blocks.append(ReductionA(384))

#         # 10 InceptionB
#         for _ in range(10):
#             blocks.append(InceptionB(1152))
#         blocks.append(ReductionB(1152))

#         # 5 InceptionC
#         for _ in range(5):
#             blocks.append(InceptionC(2144))

#         self.map = nn.Sequential(*blocks)
#         self.pool = nn.AvgPool2d(kernel_size=8)
#         self.dropout = nn.Dropout(self.dropout_prob)

#         # GradCAM
#         self.target_layers = [
#             blocks[-1],
#         ]

#         # Embedding head
#         self.embedding_head = nn.Linear(2144, self.embedding_dim, bias=True)

#         # Classfication head
#         self.classification_head = nn.Linear(2144, self.num_classes, bias=True)

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 # nn.init.constant_(m.bias, 0)

#     def set_emb_head(self, embedding_dim: int):
#         self.embedding_dim = embedding_dim
#         self.embedding_head = nn.Linear(2144, self.embedding_dim)

#     def set_cls_head(self, num_classes: int):
#         self.num_classes = num_classes
#         self.classification_head = nn.Linear(2144, self.num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.map(x)
#         x = self.pool(x)
#         x = self.dropout(x)
#         features = x.view(x.size(0), -1)

#         embeddings = self.embedding_head(features)
#         logits = self.classification_head(features)

#         return embeddings, logits

#     def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         self.eval()
#         with torch.no_grad():
#             return self.forward(x)
