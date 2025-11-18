from torch.utils.data import Dataset
from functools import lru_cache
from random import randint

from einops import repeat
import torch


class MultiPatchDataset(Dataset):
    def __init__(
        self, 
        tidewatch_dataset, 
        patch_size, 
        patch_origins, 
        num_unique_patches=None, 
        shuffle_patch=False,
        mean_inundation=None,
        std_inundation=None,
        mean_elevation=None,
        std_elevation=None
    ):
        self.tidewatch_dataset = tidewatch_dataset
        self.patch_size = patch_size
        self.num_unique_patches = num_unique_patches
        
        self.land_mask = self.tidewatch_dataset.land_mask
        
        # if patch_origins is None:
        #     self.patch_origins = self.get_land_patch_origins()
        # else:
        #     self.patch_origins = patch_origins
        self.patch_origins = patch_origins

        if self.num_unique_patches:
            self.patch_origins = self.patch_origins[:num_unique_patches]

        self.shuffle_patch = shuffle_patch

        if mean_inundation is None:
            print("Calculating mean and std inundation ...")
            self.mean_inundation, self.std_inundation = self.tidewatch_dataset.get_mean_std_inundation(
                patch_size=self.patch_size, patch_origins=self.patch_origins
            )
        else:
            self.mean_inundation, self.std_inundation = mean_inundation, std_inundation

        if mean_elevation is None:
            print("Calculating mean and std elevation ...")
            self.mean_elevation, self.std_elevation = self.tidewatch_dataset.get_mean_std_elevation(
                patch_size=self.patch_size, patch_origins=self.patch_origins
            )
        else:
            self.mean_elevation, self.std_elevation = mean_elevation, std_elevation

        self.normalized_elevation = (self.tidewatch_dataset.elevation - self.mean_elevation) / self.std_elevation

        print(f"Mean and Std Inundation: {self.mean_inundation}, {self.std_inundation}")
        print(f"Mean and Std Elevation: {self.mean_elevation}, {self.std_elevation}")
        assert self.std_inundation > 0
        assert self.std_elevation > 0
        
        # print("Dataset Length Calculation: ", len(self.patch_origins), len(self.tidewatch_dataset))
        self.number_of_patches = len(self.patch_origins)
        self.dataset_length = len(self.tidewatch_dataset) * self.number_of_patches
        
    # def get_land_patch_origins(self):
    #     """ Returns the list of patches (top-left corner) that have at-least one land cell  """
    #     patch_origins = []
    #     patch_half_size = self.patch_size # // 2
    #     for row in range(0, self.land_mask.shape[0] - self.patch_size, patch_half_size):
    #         for col in range(0, self.land_mask.shape[1] - self.patch_size, patch_half_size):
    #             patch_mask = self.land_mask[row:row+self.patch_size, col:col+self.patch_size]
    #             assert patch_mask.shape == (self.patch_size, self.patch_size)
    #             if patch_mask.sum() > self.patch_size * self.patch_size * 0.1:
    #                 patch_origins.append((row, col))
                    
    #     return patch_origins
    
    @lru_cache(maxsize=2)
    def get_tidewatch_data(self, idx):
        return self.tidewatch_dataset[idx]
    
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if idx >= self.dataset_length:
            raise IndexError("Index (%s) out of bound (%s)." % (idx, self.dataset_length))
            
        tidewatch_idx = idx // self.number_of_patches
        patch_idx = randint(0, self.number_of_patches-1) if self.shuffle_patch else idx % self.number_of_patches
            
        context, covariate, horizon = self.get_tidewatch_data(tidewatch_idx)
        
        patch_x_start, patch_y_start = self.patch_origins[patch_idx]
        patch_x_end, patch_y_end = patch_x_start + self.patch_size, patch_y_start + self.patch_size
        
        context = context[:, :, patch_x_start:patch_x_end, patch_y_start:patch_y_end]
        horizon = horizon[:, :, patch_x_start:patch_x_end, patch_y_start:patch_y_end]

        horizon_mask = ~torch.isnan(horizon)

        context = torch.nan_to_num(context, nan=self.mean_inundation)
        horizon = torch.nan_to_num(horizon, nan=self.mean_inundation)
        # Normalize
        context = (context - self.mean_inundation) / self.std_inundation
        horizon = (horizon - self.mean_inundation) / self.std_inundation

        elevation = torch.tensor(
            self.normalized_elevation[patch_x_start:patch_x_end, patch_y_start:patch_y_end]
        )
        
        # Add elevation channel in context 
        if self.tidewatch_dataset.add_elevation_channel:
            context = torch.cat(
                (context, repeat(elevation, 'h w -> d c h w', d=self.tidewatch_dataset.context_length, c=1)),
                dim=1
            )

        return context, covariate, horizon, horizon_mask
    