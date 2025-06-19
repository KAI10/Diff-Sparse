import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, reduce, rearrange


class EpsilonRelu(nn.Module):
    def __init__(self, initial_epsilon=0.0, epsilon_lb=0.0, epsilon_ub=5.0):
        super(EpsilonRelu, self).__init__()

        # self.epsilon_lb = epsilon_lb
        # self.epsilon_ub = epsilon_ub
        # self.epsilon = nn.Parameter(torch.tensor(initial_epsilon))
        self.epsilon = torch.tensor(initial_epsilon)

    def forward(self, x):
        # Clamp epsilon to the allowed range
        # epsilon_clamped = torch.clamp(self.epsilon, self.epsilon_lb, self.epsilon_ub)
        return torch.where(x > self.epsilon, x, torch.zeros_like(x))


class ConsistencyLoss:
    def __init__(self):
        self.neighbor_kernels = torch.tensor([
            [[-1, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],

            [[0, -1, 0],
             [0, 1, 0],
             [0, 0, 0]],

            [[0, 0, -1],
             [0, 1, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [-1, 1, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 1, -1],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 1, 0],
             [-1, 0, 0]],

            [[0, 0, 0],
             [0, 1, 0],
             [0, -1, 0]],

            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, -1]],
        ], dtype=torch.float32)
        self.neighbor_mask_kernels = torch.tensor([
            [[1, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 1],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [1, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 1]],
        ], dtype=torch.float32)

        self.neighbor_kernels = repeat(self.neighbor_kernels, 'o h w -> o i h w', i=1)
        self.neighbor_mask_kernels = repeat(self.neighbor_mask_kernels, 'o h w -> o i h w', i=1)

        # self.ep_relu = EpsilonRelu(initial_epsilon=5)

    def get_loss(self, inundation, elevation, mask):
        # print(f"inundation, elevation, land_mask shape: {inundation.shape}, {elevation.shape}, {land_mask.shape}")
        water_level = elevation + inundation
        # inundation_diff = F.conv2d(inundation, self.neighbor_kernels)
        water_level_diff = F.conv2d(water_level, self.neighbor_kernels)
        elevation_diff = -F.conv2d(elevation, self.neighbor_kernels)
        # print(f"inundation and elevation diff shape: {inundation_diff.shape}, {elevation_diff.shape}")

        # Only consider neighboring cells that have mask 1
        # mask dimension: batch_size x depth=1 x num_channels=1 x height x width
        mask = rearrange(mask, 'b 1 1 h w -> b h w') 
        neighbor_mask = F.conv2d(
            repeat(mask, 'b h w -> b c h w', c=1), 
            self.neighbor_mask_kernels
        )
        neighbor_count = reduce(neighbor_mask, 'b c h w -> b h w', 'sum')
        neighbor_count[neighbor_count == 0] = 1

        # Only consider land cells
        mask = repeat(mask[:, 1:-1, 1:-1], 'b h w -> b c h w', c=8)

        # inundation_diff_abs, elevation_diff_abs = torch.abs(inundation_diff), torch.abs(elevation_diff)
        # loss = (inundation_diff_abs - 3 * elevation_diff_abs) * land_mask * neighbor_mask
        # loss = F.relu(loss)

        # loss_1 = inundation_diff * elevation_diff * land_mask * neighbor_mask
        loss = water_level_diff * elevation_diff * mask * neighbor_mask
        loss = F.relu(loss)
        loss = reduce(loss, 'b c h w -> b h w', 'sum') / neighbor_count
        # loss_2 = reduce(F.relu(loss_2), 'b c h w -> b h w', 'sum') / neighbor_count
        # print(f"loss_1, loss_2: {loss_1.mean()}, {loss_2.mean()}")
        return loss.mean()  #, loss_2.mean()  # return mean over all cells
