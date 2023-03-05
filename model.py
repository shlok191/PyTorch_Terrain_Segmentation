import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


class Convolution(nn.Module):
    """2-layered convolution operations as dictated by U-Net"""

    def __init__(self, input_channels, output_channels):
        super(Convolution, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convolution(x)


class UNET(nn.Module):
    """Holds all the neural network layers required to implement U-NET"""

    def __init__(self, input_channels=3, output_channels=1, channel_count=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.initial = nn.ModuleList()
        self.final = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Implementing inital operations and reducing image dimensions for left part of "U"
        in_channels = input_channels
        for i in channel_count:
            self.initial.append(Convolution(in_channels, i))
            in_channels = i

        # Implementing final operations and reducing image dimensions for right part of "U"
        for i in reversed(channel_count):
            self.final.append(nn.ConvTranspose2d(i*2, i, kernel_size=2, stride=2))
            self.final.append(Convolution(i*2, i))
            in_channels = i

        # Defining the transition neural layers at the bottom of the U-structure
        self.transition = Convolution(channel_count[-1], channel_count[-1] * 2)
        self.final_operation = nn.Conv2d(channel_count[0], output_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []   
        
        for i in self.initial:
            x = i(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.transition(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.final), 2):
            x = self.final[i](x)

            skip_connections_2nd = skip_connections[i // 2]

            if x.shape != skip_connections_2nd.shape:
                x = tf.resize(x, size=skip_connections_2nd.shape[2:])

            concatenated = torch.cat((skip_connections_2nd, x), dim=1)
            x = self.final[i+1](concatenated)
        
        return self.final_operation(x)
