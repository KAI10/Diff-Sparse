from einops import rearrange
from torch import nn


class ConvolutionBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, num_groups):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=kernel_size)
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)
        self.silu = nn.SiLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.groupnorm(x)
        x = self.silu(x)

        x = self.conv2(x)
        x = self.groupnorm(x)
        x = self.silu(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, num_groups):
        super().__init__()

        self.conv = ConvolutionBlock(c_in, c_out, kernel_size, num_groups)
        self.pool = nn.AvgPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        return x


class PatchEmbeddingNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.d1 = DownSampleBlock(1, 16, kernel_size=(3, 3), num_groups=8)
        # self.d2 = DownSampleBlock(16, 32, kernel_size=(3, 3), num_groups=8)
        # self.d3 = DownSampleBlock(32, 64, kernel_size=(3, 3), num_groups=8)

        self.output = nn.Conv2d(16, 1, kernel_size=(1, 1))
        # self.output = nn.Conv2d(64, 1, kernel_size=(1, 1))
        self.linear = nn.Linear(config.linear_layer_size, config.spatial_embedding_size)

    def forward(self, patch_elevation):
        x = rearrange(patch_elevation, 'b h w -> b 1 h w')  # Add channel dimension
        x = self.d1(x)
        # x = self.d2(x)
        # x = self.d3(x)
        x = self.output(x)

        x = rearrange(x, 'b c h w -> b (c h w)')
        embedding = self.linear(x)
        return embedding
