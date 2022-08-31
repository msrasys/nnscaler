import torch
import torch.nn as nn
import cube


class Config:

    stages = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3]
    }

    num_layers = 50
    width_factor = 2
    num_filters = 160

    img_size = 224
    num_classes = 1024


class Bottleneck(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, width_factor: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * width_factor, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels * width_factor)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels * width_factor, out_channels * 4, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels * 4)
        
        # down sample
        self.downsample = None if in_channels == out_channels * 4 else torch.nn.ModuleList([
            nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        ])
        
        self.act3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):

        residual = x

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)

        y = self.conv3(y)
        y = self.norm3(y)

        if self.downsample is not None:
            for layer in self.downsample:
                residual = layer(residual)

        # print(residual.size(), y.size())
        y = self.act3(residual + y)
        return y

    
class WideResNet(nn.Module):

    def __init__(self):
        super().__init__()
        config = Config()

        # preprocess
        self.conv1 = nn.Conv2d(3, config.num_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(config.num_filters)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 'padding=SAME'

        self.layers = torch.nn.ModuleList([])

        stages = config.stages[config.num_layers]
        for i, block_size in enumerate(stages):
            channel = config.num_filters * (2 ** i)
            for j in range(block_size):
                if i == 0 and j == 0:
                    in_channels = channel
                elif i > 0 and j == 0:
                    in_channels = channel // 2 * 4
                else:
                    in_channels = channel * 4
                stride = 2 if i > 0 and j == 0 else 1
                print(f'add in_channel: {in_channels} | out_channel: {channel * 4}')
                block = Bottleneck(
                    in_channels, channel, config.width_factor, stride
                )
                self.layers.append(block)
        
        # postprocess
        self.fc = nn.Linear(channel * 4, config.num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, img: torch.Tensor, label: torch.Tensor):
        x = self.conv1(img)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        # print(x.size())

        for block in self.layers:
            x = block(x)

        # N C H W -> N C
        x = torch.mean(x, dim=(2,3))
        x = self.fc(x)
        loss = self.criterion(x, label)
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
