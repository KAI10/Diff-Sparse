from torch.utils.data import Dataset, DataLoader
import torch

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime, timedelta
from datastore import DataStore
from einops import repeat

from typing import Union

import pandas as pd
import numpy as np


class TidewatchTimeseriesDataset(Dataset):
    """Tidewatch timeseries dataset."""
    def __init__(
        self,
        root_dir: str,
        start_dt: datetime,
        end_dt: datetime,
        context_length: int,
        horizon_length: int,
        elevation_file: str,
        covariate_file: str,
        add_elevation_channel: bool,
        # min_max_inundation: Union[tuple[float, float], None] = None,
        # mean_std_inundation: Union[tuple[float, float], None] = None,
        transform=None
    ):
        """
            root_dir (string): Directory with all data (tidewatch frames, elevation data, covariate data).
            start_dt (datetime): Datetime for the first tidewatch frame.
            end_dt (datetime): Datetime for the last tidewatch frame.
            context_length (integer): Context length.
            horizon_length (integer): Horizon/Prediction length.
            elevation_file (string): Tiff file containing elevation data.
            covariate_file (string): CSV file containing covariate data.
            transform (callable, optional): Optional transform to be applied on each frame.
        """
        self.root_dir = root_dir
        self.start_dt = start_dt
        self.end_dt = end_dt

        self.datastore = DataStore(
            root_dir=self.root_dir,
            start_dt=self.start_dt,
            end_dt=self.end_dt,
            # mean_std_inundation=mean_std_inundation,
            covariate_file=covariate_file,
            elevation_file=elevation_file
        )

        self.context_length = context_length
        self.horizon_length = horizon_length
        self.sample_length = self.context_length + self.horizon_length

        self.td = self.end_dt - self.start_dt
        self.number_of_frames = self.td.days * 24 + self.td.seconds // 3600 + 1

        if self.number_of_frames < self.sample_length:
            raise AssertionError(
                "Not enough frames (%s) for one sample (%s)." % (self.number_of_frames, self.sample_length)
            )

        self.number_of_samples = self.number_of_frames - self.sample_length + 1

        # print("Context Length: ", self.context_length)
        # print("Horizon Length: ", self.horizon_length)
        # print("Sample Length: ", self.sample_length)
        # print("# of frames: ", self.number_of_frames)
        # print('# of samples: ', self.number_of_samples)

        # Get land mask. 1: Land, 0: water
        self.land_mask = self.get_land_mask()

        # Read elevation data
        self.elevation_file = elevation_file
        self.add_elevation_channel = add_elevation_channel
        self.elevation = self.datastore.get_elevation()

        # Read covariate data
        self.covariate_file = covariate_file
        self.covariate_df, self.normalized_covariate_df, self.covariate_scaler = self.get_covariate_dataframes()

    def get_land_mask(self):
        land_mask = self.datastore.mask
        return torch.tensor(land_mask)

    def get_covariate_dataframes(self):
        covariate_df = self.datastore.get_covariates()
        covariate_df['datetime'] = pd.to_datetime(covariate_df['datetime'])
        covariate_df = covariate_df[[
            'datetime', 'temp', 'humidity', 'precip', 'precipprob', 'windgust', 'windspeed', 'winddir',
            # 'sealevelpressure', 'solarradiation', 'solarenergy', 'severerisk'
        ]]
        # create a MinMaxScaler object
        # fit and transform the data
        # covariate_scaler = MinMaxScaler(feature_range=(-1, 1))
        covariate_scaler = StandardScaler()
        normalized_covariate_data = covariate_scaler.fit_transform(covariate_df.drop(columns=['datetime']))

        # create a new DataFrame with the normalized data
        normalized_covariate_df = pd.DataFrame(normalized_covariate_data, columns=covariate_df.columns[1:])
        normalized_covariate_df['datetime'] = covariate_df['datetime']

        normalized_covariate_df['dayofmonth'] = normalized_covariate_df['datetime'].dt.day
        # normalized_covariate_df['dayofweek'] = normalized_covariate_df['datetime'].dt.dayofweek
        normalized_covariate_df['hour'] = normalized_covariate_df['datetime'].dt.hour

        # Apply sinusoidal encoding
        normalized_covariate_df['dayofmonth_sin'] = np.sin(2 * np.pi * normalized_covariate_df['dayofmonth'] / 31)
        normalized_covariate_df['dayofmonth_cos'] = np.cos(2 * np.pi * normalized_covariate_df['dayofmonth'] / 31)
        # normalized_covariate_df['dayofweek_sin'] = np.sin(2 * np.pi * normalized_covariate_df['dayofweek'] / 7)
        # normalized_covariate_df['dayofweek_cos'] = np.cos(2 * np.pi * normalized_covariate_df['dayofweek'] / 7)
        normalized_covariate_df['hour_sin'] = np.sin(2 * np.pi * normalized_covariate_df['hour'] / 24)
        normalized_covariate_df['hour_cos'] = np.cos(2 * np.pi * normalized_covariate_df['hour'] / 24)

        normalized_covariate_df = normalized_covariate_df.drop(
            columns=['dayofmonth', 'hour'] + ['temp', 'humidity', 'precip', 'precipprob', 'windgust', 'windspeed', 'winddir']
        )

        return covariate_df.sort_values(by='datetime'), \
            normalized_covariate_df.sort_values(by='datetime'), covariate_scaler

    def get_mean_std_inundation(self, patch_size=None, patch_origins=None):
        return self.datastore.get_mean_std_inundation(patch_size, patch_origins)

    def get_mean_std_elevation(self, patch_size=None, patch_origins=None):
        return self.datastore.get_mean_std_elevation(self.elevation, patch_size, patch_origins)

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, idx):
        if idx >= self.number_of_samples:
            raise IndexError("Index (%s) out of bound (%s)." % (idx, self.number_of_samples))
            
        frames = self.datastore.datamap[idx:idx+self.context_length+self.horizon_length, :, :]
        frames = torch.tensor(frames)
        
        context = frames[0:self.context_length, :, :]
        horizon = frames[self.context_length:self.context_length+self.horizon_length, :, :]
        
        # Add channel dimension
        context = context.unsqueeze(dim=1)
        horizon = horizon.unsqueeze(dim=1)

        # Get covariate data
        covariate_start_dt = self.start_dt + timedelta(hours=idx)
        covariate_end_dt = covariate_start_dt + timedelta(hours=self.sample_length)
        covariate_mask = (self.normalized_covariate_df['datetime'] >= covariate_start_dt) & \
                         (self.normalized_covariate_df['datetime'] < covariate_end_dt)
        covariate = self.normalized_covariate_df.loc[covariate_mask].drop(columns=['datetime']).to_numpy()
        covariate = torch.tensor(covariate, dtype=torch.float32)

        return context, covariate, horizon
        # return context, self.normalized_elevation_frame.clone().detach(), covariate, horizon
