from sklearn.preprocessing import MinMaxScaler, StandardScaler

from functools import lru_cache
from itertools import product
from datetime import timedelta, datetime
from typing import Union

import pandas as pd
import numpy as np
import rasterio
import glob
import json
import os


class DataStore:
    def __init__(
        self,
        root_dir: str,
        start_dt: datetime,
        end_dt: datetime,
        covariate_file: str,
        elevation_file: str
    ):
        self.root_dir = root_dir
        self.start_dt = start_dt
        self.end_dt = end_dt

        self.td = self.end_dt - self.start_dt
        self.number_of_frames = self.td.days * 24 + self.td.seconds // 3600 + 1
        
        self.mask = self.get_mask()
        self.height, self.width = self.mask.shape
        
        self.datamap_name = f"{self.start_dt.strftime('%m-%d-%Y-%H:%M:%S')}_to_" + \
                            f"{self.end_dt.strftime('%m-%d-%Y-%H:%M:%S')}.memmap"
        self.datamap_path = f"{self.root_dir}/{self.datamap_name}"

        if not os.path.isfile(self.datamap_path):
            print(f"Creating datamap at {self.datamap_path} ...")
            self.datamap = np.memmap(
                self.datamap_path,
                dtype='float32', 
                mode='w+',
                shape=(self.number_of_frames, self.height, self.width)
            )
            self.save_datamap()
        else:        
            self.datamap = np.memmap(
                self.datamap_path,
                dtype='float32', 
                mode='r',
                shape=(self.number_of_frames, self.height, self.width)
            )
        
        print(f"datamap shape: {self.datamap.shape}")
        
        self.covariate_file = covariate_file
        self.elevation_file = elevation_file
        
    def save_datamap(self):
        fetch_dt, frame_index = self.start_dt, 0
        while fetch_dt <= self.end_dt:
            print(str(fetch_dt), end='\r')
            key = fetch_dt.strftime("%m_%d_%Y_%H_%M_%S")
            value = rasterio.open(f"{self.root_dir}/{key}.tif").read(1)
            value = np.where(value >= -3e+38, value, np.nan)
            self.datamap[frame_index, :, :] = value
            fetch_dt = fetch_dt + timedelta(hours=1)
            frame_index += 1

        print("Flushing the datamap")
        self.datamap.flush()

    @lru_cache(maxsize=64)
    def get(self, key):
        value = rasterio.open(f"{self.root_dir}/{key}.tif").read(1)  #
        # Replace garbage (inundation on water body) inundation value with max_inundation
        # value[value < -3e+38] = self.mean_inundation
        # normalized_value = self.inundation_scaler.transform(value.reshape(-1, 1)).reshape(value.shape)
        # return normalized_value
        value = np.where(value >= -3e+38, value, np.nan)
        return value

    def get_mask(self):
        mask = rasterio.open(
            f"{self.root_dir}/{self.start_dt.strftime('%m_%d_%Y_%H_%M_%S')}.tif"
        ).read_masks(1)
        mask[mask > 1] = 1
        return mask

    def get_elevation(self):
        elevation_frame = rasterio.open(f"{self.root_dir}/{self.elevation_file}").read(1)
        # Replace garbage elevation values with 0 (elevation 0 at sea level)
        # elevation_frame[elevation_frame < -3e+38] = 0
        elevation_frame = np.where(elevation_frame >= -3e+38, elevation_frame, 0)
        return elevation_frame
    
    def get_covariates(self):
        covariates_df = pd.concat(map(
            pd.read_csv, 
            glob.glob(f"{self.root_dir}/{self.covariate_file}")
        ))
        return covariates_df
        # return pd.read_csv('%s/%s' % (self.root_dir, self.covariate_file))

    def get_mean_std_elevation(self, elevation, patch_size=None, patch_origins=None):
        if patch_size is None:
            return np.nanmean(elevation), np.nanstd(elevation)

        patch_elevations = []
        for i, (x, y) in enumerate(patch_origins):
            patch_elevation = elevation[x:x+patch_size, y:y+patch_size]
            patch_elevations.append(patch_elevation)

        patch_elevations = np.vstack(patch_elevations)
        return np.nanmean(patch_elevations), np.nanstd(patch_elevations)

    def get_mean_std_inundation(self, patch_size=None, patch_origins=None):
        if patch_size is None:
            return np.nanmean(self.datamap), np.nanstd(self.datamap)

        num_patches = len(patch_origins)
        datamap_stats_name = f"{self.start_dt.strftime('%m-%d-%Y-%H:%M:%S')}_to_" + \
            f"{self.end_dt.strftime('%m-%d-%Y-%H:%M:%S')}_{patch_size}_{num_patches}.json"

        if os.path.exists(f"{self.root_dir}/{datamap_stats_name}"):
            datamap_stats = json.load(open(f"{self.root_dir}/{datamap_stats_name}", "r"))
            return datamap_stats["mean_inundation"], datamap_stats["std_inundation"]

        print("Calculating mean inundation ...")
        sum_inundation, non_nan_count = 0, 0
        for i, (x, y) in enumerate(patch_origins):
            print(f"Processing patch : {i:2d}", end='\r', flush=True)
            patch_data = self.datamap[:, x:x+patch_size, y:y+patch_size]
            sum_inundation += np.nansum(patch_data)
            non_nan_count += np.count_nonzero(~np.isnan(patch_data))

        mean_inundation = sum_inundation / non_nan_count

        print("Calculating std inundation ...")
        square_sum_inundation = 0
        for i, (x, y) in enumerate(patch_origins):
            print(f"Processing patch : {i:2d}", end='\r', flush=True)
            patch_data = self.datamap[:, x:x+patch_size, y:y+patch_size]
            patch_data = np.nan_to_num(patch_data, nan=0)
            square_sum_inundation += np.sum(np.square(patch_data - mean_inundation))

        std_inundation = np.sqrt(square_sum_inundation / non_nan_count)

        datamap_stats = {
            "mean_inundation": float(mean_inundation),
            "std_inundation": float(std_inundation)
        }
        json.dump(datamap_stats, open(f"{self.root_dir}/{datamap_stats_name}", "w"))
        return mean_inundation, std_inundation
        
