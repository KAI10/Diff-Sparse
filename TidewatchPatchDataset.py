from torch.utils.data import Dataset
from functools import lru_cache
from random import randint

from einops import repeat
import torch

import numpy as np


class MultiPatchDataset(Dataset):
    def __init__(
        self, 
        tidewatch_dataset, 
        patch_origins, 
        config,
        masks=None,
        shuffle_patch=False,
        mean_inundation=None, std_inundation=None,
        mean_elevation=None, std_elevation=None
    ):
        self.tidewatch_dataset = tidewatch_dataset
        # if patch_origins is None:
        #     self.patch_origins = self.get_land_patch_origins()
        # else:
        #     self.patch_origins = patch_origins
        self.patch_origins = patch_origins
        self.config = config
        self.masks = masks
        self.shuffle_patch = shuffle_patch

        self.patch_size = config.patch_size
        self.num_unique_patches = config.num_unique_patches
        self.patch_origins = self.patch_origins[:self.num_unique_patches]

        if masks is not None:
            self.num_masks = config.num_test_masks
            self.masks = self.masks[:self.num_masks, :, :]
        else:
            self.num_masks = None

        self.land_mask = self.tidewatch_dataset.land_mask
        
        self.add_elevation_channel = config.add_elevation_channel
        self.add_data_mask_channel = config.add_data_mask_channel
        self.data_missing_percentage = config.data_missing_percentage

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
        # print(f"horizon, horizon_mask shape: {horizon.shape}, {horizon_mask.shape}")

        context = torch.nan_to_num(context, nan=self.mean_inundation)
        horizon = torch.nan_to_num(horizon, nan=self.mean_inundation)
        # Normalize
        context = (context - self.mean_inundation) / self.std_inundation
        horizon = (horizon - self.mean_inundation) / self.std_inundation

        elevation = torch.tensor(
            self.normalized_elevation[patch_x_start:patch_x_end, patch_y_start:patch_y_end]
        )

        if self.add_data_mask_channel:
            if self.num_masks is None: 
                # For trainging and validation, randomly generate mask
                data_mask = self.generate_data_mask(horizon_mask[0,0,:,:], self.data_missing_percentage)
            else:
                # For test, choose from pre-selected masks
                mask_idx = tidewatch_idx % self.num_masks
                # Multiply by land_mask to ensure that only land_cells can be 1
                data_mask = self.masks[mask_idx] * horizon_mask[0,0,:,:]
            
            # # Placing 0 in-place of missing data
            # context = context * data_mask
            data_mask_expanded = repeat(data_mask, "h w -> d c h w", d=context.shape[0], c=context.shape[1])
            
            # # Place zero where there is no sensor
            # context = context * data_mask_expanded
            
            noise_mask = torch.randn(*data_mask_expanded.shape)
            # Place random gaussian noise at the missing data
            context = context * data_mask_expanded + ~data_mask_expanded * noise_mask
        
        # Add elevation channel in context 
        if self.add_elevation_channel:
            context = torch.cat(
                (context, repeat(elevation, 'h w -> d c h w', d=self.config.context_length, c=1)),
                dim=1
            )
        if self.add_data_mask_channel:
            context = torch.cat(
                (context, repeat(data_mask, 'h w -> d c h w', d=self.config.context_length, c=1)),
                dim=1
            )

        return context, covariate, horizon, horizon_mask

    
    def generate_data_mask(self, land_mask, p):
        """
        Generates a missing data mask M using a probabilistic approach.
        For each land cell, it is marked as missing with the given probability.

        M[i, j] = 1 if land_mask[i, j] is land and the cell is 'observed'.
        M[i, j] = 0 if land_mask[i, j] is land and the cell is stochastically 'missing'.
        M[i, j] = 0 if land_mask[i, j] is sea.

        Args:
            land_mask: A (D, D) numpy array where 1=land, 0=sea.
            p: The probability (0.0 to 1.0) that any given
                                    *land* cell will be marked as missing.

        Returns:
            A (D, D) numpy array (int8) representing the missing data mask M.
        """
        assert 0 <= p <= 1

        # Create a random matrix of the same shape as land_mask with values in [0, 1)
        random_chance_matrix = torch.rand(*land_mask.shape)

        # Initialize the missing mask:
        # - Sea cells (land_mask == 0) are always 0 (missing/irrelevant).
        # - Land cells (land_mask == 1) are initially considered observed (1).
        data_mask = land_mask.clone().detach()

        # Identify land cells that should be masked based on the probability
        # Condition for masking a land cell: (it is land) AND (its random chance is less than the probability)
        condition_to_mask_land_cell = (land_mask == 1) & (random_chance_matrix < p)

        # Set these selected land cells to 0 (missing)
        data_mask[condition_to_mask_land_cell] = 0
        # print(f"Percentage masked: {data_mask.sum()*100 / land_mask.sum()}")

        return data_mask