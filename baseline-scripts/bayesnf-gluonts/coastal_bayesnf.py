import warnings
warnings.simplefilter('ignore')

#import contextily as ctx
import geopandas as gpd
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import random
from bayesnf.spatiotemporal import BayesianNeuralFieldMAP
import time
import properscoring as ps
from einops import rearrange

#from cartopy import crs as ccrs
from shapely.geometry import Point
from mpl_toolkits.axes_grid1 import make_axes_locatable

patch_size = int(sys.argv[1])
num_patches = int(sys.argv[2])
SEED = int(sys.argv[3])

print('patch size',patch_size)
print('num_patches',num_patches)
print('seed',SEED,flush=True)

data_dir = f'/project/biocomplexity/Diff_Spatio_Temporal_Grid/Datasets/bayesnf_datasets/'

df_train = pd.read_parquet(f'{data_dir}train/tidewatch_{patch_size}_{num_patches}.pq')
#df_train.dtypes

## masking
# Mask = 0.95
# #SEED = 0
# random_indices = df_train.sample(frac=Mask, random_state=SEED).index
# df_train.loc[random_indices, 'inundation'] = np.nan



model = BayesianNeuralFieldMAP(
  width=128,
  depth=2,
  freq='H',
  seasonality_periods=['D', 'W'], # Daily and weekly seasonality, same as [24, 24*7]
  num_seasonal_harmonics=[4, 4], # Four harmonics for each seasonal factor.
  feature_cols=['datetime', 'latitude', 'longitude','elevation'], # time, spatial 1, ..., spatial n
  target_col='inundation',
  observation_model='NORMAL',
  timetype='index',
  standardize=['latitude', 'longitude','elevation'],
)

#np.random.seed(SEED)
#random.seed(SEED)

train_time_start = time.time()

model = model.fit(
    df_train,
    seed=jax.random.PRNGKey(SEED),
    ensemble_size=8,
    batch_size=32,
    num_epochs=40,
)

print('train done in',time.time()-train_time_start,'seconds',flush=True)

s = 0
for p in model.params_:
    s = s+p.flatten().shape[0]
print('model params:',s,flush=True)

all_crps = []
all_crps_denominator = []
all_rmse = []
all_rmse_denom = []

mx = -1e9
mn = 1e9

nanmean = np.nanmean(df_train['inundation'].values)

#np.random.seed(SEED)
#random.seed(SEED)

inference_time_start = time.time()

for window in range(0,168-12+1):
    
    if window%10==0:
        print(window,flush=True)

    df_test = pd.read_parquet(f'{data_dir}test/tidewatch_{patch_size}_{num_patches}_window_{window}.pq')
    
    if 'landmask' in df_test.columns.tolist():
        
        if window%10==0:
            print('before removing non-land cells test data size for window',window,'is',df_test.shape[0],flush=True)
        
        df_test = df_test[df_test.landmask==1]
        df_test = df_test.reset_index()
        
        if window%10==0:
            print('after removing non-land cells test data size for window',window,'is',df_test.shape[0],flush=True)
    
    if patch_size==16 and num_patches==2:
        df_test = df_test[0:-1]
    
    df_test['latitude'] = df_test['latitude'].astype('float')
    df_test['longitude'] = df_test['longitude'].astype('float')
    #df_test['inundation'] = df_test['inundation'].fillna(nanmean)
    df_test = df_test.dropna(subset=['inundation'])
    yhat1, yhat_quantiles1 = model.predict(df_test.drop(columns={'inundation'}))




    all_forecasts = rearrange(yhat1[0],'s v -> v s')
    all_truths = df_test['inundation'].values
    crps = ps.crps_ensemble(all_truths, all_forecasts)
    
    all_crps.append(crps.sum())
    all_crps_denominator.append(np.sum(np.absolute(all_truths)))
    #crps_land_only = ps.crps_ensemble(all_truths_land_only, all_forecasts_land_only)

    #val_loss = crps.sum()/np.sum(np.absolute(all_truths)) # https://arxiv.org/pdf/2401.03006
    #val_loss_land_only = crps_land_only.sum()/np.sum(np.absolute(all_truths_land_only))

    all_forecasts_average = np.mean(all_forecasts,axis=-1)
    rmse = np.sum((all_forecasts_average-all_truths)*(all_forecasts_average-all_truths))
    all_rmse.append(rmse)
    all_rmse_denom.append(all_truths.shape[0])
    
    mx = max(mx,np.max(all_truths))
    mn = min(mn,np.min(all_truths))
    # all_forecasts_average_land_only = np.mean(all_forecasts_land_only,axis=-1)
    # rmse_land_only = (np.mean((all_forecasts_average_land_only-all_truths_land_only)*(all_forecasts_average_land_only-all_truths_land_only)))**0.5
    # rmse_land_only = rmse_land_only/(np.max(all_truths_land_only)-np.min(all_truths_land_only))

    # return val_loss,rmse,val_loss_land_only,rmse_land_only

    #all_crps.append()
    
nacrps = sum(all_crps)/sum(all_crps_denominator)
nrmse = ((sum(all_rmse)/sum(all_rmse_denom))**0.5)/(mx - mn)

print('nacrps:',nacrps)
print('nrmse:',nrmse)

print('inference done in',time.time()-inference_time_start,'seconds')
