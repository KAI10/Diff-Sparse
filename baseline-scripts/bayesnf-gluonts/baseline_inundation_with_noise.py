import warnings
import sys
warnings.filterwarnings('ignore')
import numpy as np
np.bool = np.bool_  ## https://stackoverflow.com/a/76224186 - dirty trick to handle mxnet vs numpy version issue
import torch
import random
import time

import matplotlib.pyplot as plt
from gluonts.dataset.util import to_pandas

import gc
from training_config import TrainingConfig
from baseline_utils import *
from einops import repeat,rearrange

from sklearn.preprocessing import MinMaxScaler
from gluonts.transform import MeanValueImputation,CausalMeanValueImputation

import rasterio
import pickle

from lightning_training import get_train_dataset, get_validation_dataset

config = TrainingConfig()

SEED = int(sys.argv[1])
config.patch_size = int(sys.argv[2])
config.num_unique_patches = int(sys.argv[3])
model_name = sys.argv[4]
lstnet_rnn_layer = 3
config.num_epochs = 40
context_length = 12
prediction_horizon = 12
no_windows = 24*7-prediction_horizon+1
#no_windows = 24
config.train_batch_size = 32
num_scenarios = 8
FORECAST_TYPE = 'GPU'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print('current configuration:')
print('SEED:',SEED)
print('patch size:',config.patch_size)
print('number of patches:',config.num_unique_patches)
print('model:',model_name)
print('number of rnn layers for lstnet:',lstnet_rnn_layer)
print('epochs:',config.num_epochs)
print('context length:',context_length)
print('prediction horizon length:',prediction_horizon)
print('number of windows for prediction:',no_windows)
print('batch size during training:',config.train_batch_size)
print('number of scenarios to generate:',num_scenarios)
print('',flush=True)

patch_origins = pickle.load(open(f'/project/biocomplexity/Diff_Spatio_Temporal_Grid/Datasets/non_overlapping_patches_{config.patch_size}.pickle','rb'))
print(patch_origins)

processed_origins = [(x,y) for (x,y,z) in patch_origins[0:config.num_unique_patches]]
#processed_origins = [()]
print(processed_origins)

other_data, train_data = get_train_dataset(config,processed_origins)
valid_data = get_validation_dataset(config,processed_origins,train_data)

# processed_origins = all_origin[7:8] if config.num_unique_patches==1 and config.patch_size==128 else all_origin[0:config.num_unique_patches]
# print(processed_origins,flush=True)



tot_patch_row = 4392
tot_patch_col = 2889
tot_train_time = 1956
tot_valid_time = 168+12
#tot_train_time = 36
#tot_valid_time = 16
tidewatch_train_data = np.memmap(filename='/project/biocomplexity/Diff_Spatio_Temporal_Grid/Datasets/Tiff/03-28-2024-06:00:00_to_06-17-2024-17:00:00.memmap',
                       mode='r',shape=(tot_train_time,tot_patch_row,tot_patch_col),dtype='float32')

tidewatch_validation_data = np.memmap(filename='/project/biocomplexity/Diff_Spatio_Temporal_Grid/Datasets/Tiff/06-17-2024-06:00:00_to_06-24-2024-17:00:00.memmap',
                       mode='r',shape=(tot_valid_time,tot_patch_row,tot_patch_col),dtype='float32')


#tidewatch_test_data = np.vstack((tidewatch_train_data,tidewatch_validation_data[12:,:,:]))
print('data has been read',flush=True)

elevation_frame_read = rasterio.open('/project/biocomplexity/Diff_Spatio_Temporal_Grid/Datasets/Tiff/ESVA_Elevation_EPSG_3857_USGS30m.tif')
elevation_frame = elevation_frame_read.read(1)
elevation_frame[elevation_frame < -3e+38] = 0

all_patch_data = []
for o in processed_origins:
    cur_patch_data = np.vstack((tidewatch_train_data[:,o[0]:o[0]+config.patch_size,o[1]:o[1]+config.patch_size],
                              tidewatch_validation_data[12:,o[0]:o[0]+config.patch_size,o[1]:o[1]+config.patch_size])) 
    print(o,cur_patch_data.shape)
    all_patch_data.append(cur_patch_data)

patch_data = np.array(all_patch_data)
nan_replace_train = np.nanmean(patch_data[:,0:tot_train_time,:,:])
print(nan_replace_train)

new_mask_data = get_mask_data_from_datamap(processed_origins,tidewatch_train_data,tidewatch_validation_data,config)
new_mask_data_reshaped = rearrange(new_mask_data,'p t h w -> p (h w) t')
new_mask_data_reshaped = new_mask_data_reshaped[:,:,tot_train_time:]

print('tot train time',tot_train_time,'# windows',no_windows)
print('before stacking mask data shape is: ',new_mask_data_reshaped.shape)

stacked_mask = []
for i in range(0,no_windows):
    stacked_mask.append(new_mask_data_reshaped[:,:,i:i+prediction_horizon])

new_mask_data_reshaped = np.array(stacked_mask)
new_mask_data_reshaped_1 = rearrange(new_mask_data_reshaped,'t p v h -> (p t) v h')
#new_mask_data_reshaped_2 = rearrange(new_mask_data_reshaped,'t p v h -> p t v h')
#new_mask_data_reshaped_2 = rearrange(new_mask_data_reshaped_2,'a b c d ->(a b) c d')
print('mask shape for evaluation',new_mask_data_reshaped_1.shape)
#np.save(f'sample_data/mask-D-{config.patch_size}-K-{config.num_unique_patches}-W-{no_windows}-H-{prediction_horizon}.npy',new_mask_data_reshaped_1)



train_df,test_df,uni_train_ds,uni_test_ds,all_patch_df,all_elev = create_univariate_data(processed_origins,tidewatch_train_data,tidewatch_validation_data, 
                                                                                         elevation_frame,config,tot_train_time,nan_replace=nan_replace_train)

def create_multivariate_data(all_patch_df,all_elev_data):
    #multivariate_ts = univariate_df.values
    multivariate_ds = ListDataset( [ {'start':"2024-03-28 06:00:00", 'target': (univariate_df.values).T,'patch':'patch_'+str(idx).zfill(3),
                                      'feat_static_real':elev_info } for idx,(univariate_df,elev_info) in enumerate(zip(all_patch_df,all_elev_data))] , 
                                  freq="H", one_dim_target=False)
    return multivariate_ds

def create_multivariate_data_new(all_patch_df,all_elev_data, starting_hour=6, per_timestep_hr = 1):
    #multivariate_ts = univariate_df.values
    hourly_data = np.array([((starting_hour+cur_time*per_timestep_hr)%24)/24 for cur_time in range(0,all_patch_df[0].shape[0])])
    hourly_data_for_gluonts = repeat(hourly_data,'t -> f t', f=1)
    print('feat_dynamic_real shape for one patch',hourly_data_for_gluonts.shape)
    multivariate_ds = ListDataset( [ {'start':"2024-03-28 06:00:00", 'target': (univariate_df.values).T,'patch':'patch_'+str(idx).zfill(3),
                                      'feat_dynamic_real': hourly_data_for_gluonts,'feat_dynamic_cat': hourly_data_for_gluonts} 
                                    for idx,(univariate_df,elev_info) in enumerate(zip(all_patch_df,all_elev_data))], 
                                  freq="H", one_dim_target=False)
    return multivariate_ds

#if model_name=='DVAR':
#    multivariate_ds = create_multivariate_data_new(all_patch_df,all_elev)
#else:
multivariate_ds = create_multivariate_data(all_patch_df,all_elev)

univariate_training_data, univariate_test_pairs = create_test_pairs(uni_test_ds,prediction_horizon,no_windows,
                                                                    "2024-06-17 17:00:00")
multivariate_training_data, multivariate_test_pairs = create_test_pairs(multivariate_ds,prediction_horizon,no_windows,
                                                                        "2024-06-17 17:00:00")

predictor = train(univariate_training_data,multivariate_training_data,
                  prediction_horizon,context_length,config,model_name,MP=0.95)

print('starting forecast',flush=True)
forecast_start = time.time()
model_type='multivariate' if model_name in['LSTNET','TIMEGRAD','DVAR','GPVAR'] else 'univariate' 
my_test_pair = multivariate_test_pairs if model_type=='multivariate' else univariate_test_pairs
forecasts = get_forecasts(my_test_pair,predictor,num_scenarios)
forecast_end = time.time()
print('forecast gathered in',forecast_end-forecast_start,'seconds',flush=True)

relevant_test_pair = list(my_test_pair)
all_truths = np.array([test_pair[1]['target'] for test_pair in relevant_test_pair])

new_mask_data_reshaped = rearrange(new_mask_data,'p t h w -> p (h w) t')
new_mask_data_reshaped = new_mask_data_reshaped[:,:,tot_train_time:]

stacked_mask = []
for i in range(0,no_windows):
    stacked_mask.append(new_mask_data_reshaped[:,:,i:i+prediction_horizon])

new_mask_data_reshaped = np.array(stacked_mask)
new_mask_data_reshaped_1 = rearrange(new_mask_data_reshaped,'t p v h -> (p t) v h')
#new_mask_data_reshaped_2 = rearrange(new_mask_data_reshaped,'t p v h -> p t v h')
#new_mask_data_reshaped_2 = rearrange(new_mask_data_reshaped_2,'a b c d ->(a b) c d')
print('mask shape for evaluation',new_mask_data_reshaped_1.shape)

#crps,rmse,crps_mask,rmse_mask = get_crps_naive(forecasts,my_test_pair,gluonts_mask_data_repeated,model_type) #sff
##previous evaluation
crps,rmse,crps_mask,rmse_mask = get_crps_v3(all_truths,forecasts,new_mask_data_reshaped_1)
print('current configuration:')
print('SEED:',SEED)
print('patch size:',config.patch_size)
print('number of patches:',config.num_unique_patches)
print('model:',model_name)
print('number of rnn layers for lstnet:',lstnet_rnn_layer)
print('epochs:',config.num_epochs)
print('context length:',context_length)
print('prediction horizon length:',prediction_horizon)
print('number of windows for prediction:',no_windows)
print('batch size during training:',config.train_batch_size)
print('number of scenarios to generate:',num_scenarios)
print('forecast type:',FORECAST_TYPE)
print('NACRPS loss:',crps,flush=True)
print('RMSE loss:',rmse,flush=True)
print('masked NACRPS loss:',crps_mask,flush=True)
print('masked RMSE loss:',rmse_mask,flush=True)
