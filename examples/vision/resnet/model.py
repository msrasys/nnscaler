from typing import List, Optional, Callable
import torch
import torch.nn as nn
import cube


class Config:

    width_factor = 1  # for scaling default 1
    inplanes = 160 # 64
    # setting for wide-resnet 50
    layers : List[int] = [3, 4, 6, 3]

    # setting for wide-resnet 101
    layers : List[int] = [3, 4, 23, 3]

    width_per_group = 128 * width_factor
    # conv2d:
    #   in_channel: 128  | out_channel: 128  | stride: 1 | groups: 1 | dilation: 1
    #   in_channel: 256  | out_channel: 256  | stride: 2 | groups: 1 | dilation: 1
    #   in_channel: 512  | out_channel: 512  | stride: 1 | groups: 1 | dilation: 1
    #   in_channel: 1024 | out_channel: 1024 | stride: 2 | groups: 1 | dilation: 1
    # conv2d inputs:
    #   torch.Size([1, 128, 128, 128])
    #   torch.Size([1, 256, 64, 64])
    #   torch.Size([1, 512, 32, 32])
    #   torch.Size([1, 1024, 16, 16])

    # input
    img_size = 224
    num_classes = 1024 # 1000


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    # print(f'conv2d: in_channel: {in_planes} | out_channel: {out_planes} | stride: {stride} | groups: {groups} | dilation: {dilation}')
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        print(f'adding conv2d  channel: {width}, stride: {stride}, padding: {dilation}')
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # print(f'conv2d input shape: {out.size()}')
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        print(identity.size(), out.size())
        out += identity
        out = self.relu(out)

        return out


class WideResNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layers = Config.layers
        self.num_classes = 1000
        self._norm_layer = nn.BatchNorm2d
        self.block = Bottleneck
        self.inplanes = 64
        self.dilation = 1
        self.replace_stride_with_dilation = [False, False, False]
        self.groups = 1
        self.base_width = Config.width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64,  self.layers[0])
        self.layer2 = self._make_layer(128, self.layers[1], stride=2, dilate=self.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, self.layers[2], stride=2, dilate=self.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, self.layers[3], stride=2, dilate=self.replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, self.num_classes)
        self.loss_func = nn.CrossEntropyLoss()
    
    def _make_layer(self, planes: int, blocks: int, stride: int = 1, dilate = False):
        block = Bottleneck

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            x = layer(x)
        for layer in self.layer3:
            x = layer(x)
        for layer in self.layer4:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        loss = self.loss_func(x, target)
        return loss


class ImageDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int):

        self.bs = batch_size
        self.img_size = Config.img_size
        self.num_classes = Config.num_classes
        super().__init__(
            shapes=([batch_size, 3, self.img_size, self.img_size,],
                    [batch_size],
            ),
            dtypes=(torch.float, torch.int),
            batch_dims=(0, 0)
        )
        self.samples = [self.random_sample()]
        
    def random_sample(self):
        img = torch.rand(
            *(self.bs, 3, self.img_size, self.img_size),
            dtype=torch.float,
            device=torch.cuda.current_device()
        )
        labels = torch.randint(
            0, self.num_classes,
            size=(self.bs,),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        return (img, labels)
    
    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]