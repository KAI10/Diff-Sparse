# Define the hidden state net
import torch

from einops import rearrange
from torch import nn


class ConvolutionBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, num_groups):
        super().__init__()
        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=kernel_size)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=kernel_size)
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
        self.pool = nn.AvgPool3d((1, 2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        return x
    
    
class HiddenStateNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_embedding_size = config.spatial_embedding_size
        self.linear_layer_size = config.linear_layer_size

        if config.use_covariate_embedding:
            self.linear_layer_size += config.covariate_dimension

        # self.input_channel = 2 if config.add_elevation_channel else 1
        self.input_channel = config.num_channels

        # self.d1 = DownSampleBlock(self.input_channel, 8, kernel_size=(1, 3, 3), num_groups=8) # For patch size 16
        self.d1 = DownSampleBlock(self.input_channel, 16, kernel_size=(1, 3, 3), num_groups=8)
        self.d2 = DownSampleBlock(16, 32, kernel_size=(1, 3, 3), num_groups=8)
        self.d3 = DownSampleBlock(32, 64, kernel_size=(1, 3, 3), num_groups=8)
        
        # self.output = nn.Conv3d(8, 1, kernel_size=(1, 1, 1)) # For patch size 16
        self.output = nn.Conv3d(64, 1, kernel_size=(1, 1, 1))
        self.linear = nn.Linear(self.linear_layer_size, self.spatial_embedding_size)
        
    def forward(self, context, covariate):
        # context: batch of context (batch_size x context_length x num_channels=2 x height x width)
        # covariate: batch of covariate (batch_size x context_length x covariate_dimension)
        # if not self.config.add_elevation_channel:
        #     context = context[:, :, :1, :, :]  # Removing elevation channel

        reshaped_context = rearrange(context, 'b d c h w -> b c d h w')
        
        x = self.d1(reshaped_context)
        x = self.d2(x)
        x = self.d3(x)
        x = self.output(x)
        
        x = rearrange(x, 'b c d h w -> b d (c h w)')

        if self.config.use_covariate_embedding:
            # print(f"x.shape: {x.shape}")
            # print(f"covariate.shape: {covariate.shape}")
            x = torch.cat((x, covariate), dim=2)
        
        embedding = self.linear(x)
        return embedding
