from torch import nn


class SRCNN(nn.Module):
    def __init__(self, cfg, num_channels=1):
        super(SRCNN, self).__init__()

        base_channels = cfg["BASE_CHANNELS"]
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(
            num_channels, 2 * base_channels, kernel_size=9, padding=9 // 2
        )
        self.conv2 = nn.Conv2d(
            2 * base_channels, base_channels, kernel_size=5, padding=5 // 2
        )
        self.conv3 = nn.Conv2d(
            base_channels, num_channels, kernel_size=5, padding=5 // 2
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First upsample the image with bicubic interpolation
        x = self.upsample(x)
        # Then 1) path extraction and representation
        x = self.relu(self.conv1(x))
        # Then 2) non-linear-mapping
        x = self.relu(self.conv2(x))
        # Then 3) reconstruction
        x = self.conv3(x)
        return x


def conv_relu(chan):
    return [nn.Conv2d(chan, chan, kernel_size=3, padding=1), nn.ReLU()]


class SRCNN2(nn.Module):
    def __init__(self, cfg, num_channels=1, depth=1):
        super().__init__()

        base_channels = cfg["BASE_CHANNELS"]
        depth = cfg["DEPTH"]

        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(num_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        for _ in range(depth):
            layers.extend(conv_relu(base_channels))
        layers.extend(
            [
                nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            ]
        )
        for _ in range(depth):
            layers.extend(conv_relu(2 * base_channels))
        layers.append(
            nn.Conv2d(2 * base_channels, num_channels, kernel_size=3, padding=1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
