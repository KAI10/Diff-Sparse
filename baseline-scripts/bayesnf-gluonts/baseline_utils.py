# # Import of libraries

import numpy as np
from einops import rearrange,reduce,repeat
import pandas as pd
from gluonts.dataset.common import ListDataset
from datetime import datetime
from gluonts.mx import DeepAREstimator, DeepVAREstimator, GPVAREstimator, SimpleFeedForwardEstimator, Trainer
from gluonts.mx.model.lstnet import LSTNetEstimator
#from gluonts.mx.model.tft import TemporalFusionTransformerEstimator
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
#import mxnet as mx
#from gluonts.mx.model.lstnet import LSTNetEstimator
from training_config import TrainingConfig
from gluonts.evaluation import Evaluator
import properscoring as ps

from gluonts.transform import MeanValueImputation,CausalMeanValueImputation

from diffusers import (
    PNDMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    DEISMultistepScheduler,
)

from pts.model.time_grad import TimeGradEstimator
from pts.dataset.repository.datasets import dataset_recipes

## creates patch data for gluonts pandasdataset
## returns a 2D dataframe corresponding to time series inundation of grids of each patch
## rows are time indexes starting from config.train_start_datetime, increments hourly
## each column is named patch_{index}_cell_{c}, representing the time series inundation of cell c of index-th patch
## no spatial correlation maintained
def prepare_patch_data_for_gluonts_old(patch_row, patch_col, sample_data,config,patch_idx,nan_replace=None):
    #config.patch_size = 12
    cur_patch = sample_data[:,patch_row:patch_row+config.patch_size,patch_col:patch_col+config.patch_size]
    if nan_replace:
        cur_patch = np.nan_to_num(cur_patch, nan=nan_replace)
    #cur_mask = land_mask[patch_row:patch_row+config.patch_size,patch_col:patch_col+config.patch_size]
    cur_patch = rearrange(cur_patch, 't h w -> t (h w)')
    #cur_mask = rerrange(cur_mask,'h w -> (h w)')
    cur_patch_df =  pd.DataFrame(cur_patch)
    cur_patch_df = cur_patch_df.set_index(pd.date_range(config.train_start_datetime, periods=cur_patch_df.shape[0], freq='H'))
    cur_patch_df.columns = ['patch_'+str(patch_idx).zfill(3)+'_cell_'+str(col).zfill(5) for col in cur_patch_df.columns]
    return cur_patch_df#,cur_mask

## memory efficient
def prepare_patch_data_for_gluonts(patch_row, patch_col, sample_data_train,config,patch_idx,nan_replace=None,overlap=12,sample_data_valid=None):
    #config.patch_size = 12
    if sample_data_valid is not None:
        cur_patch = np.vstack((sample_data_train[:,patch_row:patch_row+config.patch_size,patch_col:patch_col+config.patch_size],
                              sample_data_valid[12:,patch_row:patch_row+config.patch_size,patch_col:patch_col+config.patch_size]))
    else:
        cur_patch = sample_data_train[:,patch_row:patch_row+config.patch_size,patch_col:patch_col+config.patch_size]
    if nan_replace:
        cur_patch = np.nan_to_num(cur_patch, nan=nan_replace)
    #cur_mask = land_mask[patch_row:patch_row+config.patch_size,patch_col:patch_col+config.patch_size]
    cur_patch = rearrange(cur_patch, 't h w -> t (h w)')
    #cur_mask = rerrange(cur_mask,'h w -> (h w)')
    cur_patch_df =  pd.DataFrame(cur_patch)
    cur_patch_df = cur_patch_df.set_index(pd.date_range(config.train_start_datetime, periods=cur_patch_df.shape[0], freq='H'))
    cur_patch_df.columns = ['patch_'+str(patch_idx).zfill(3)+'_cell_'+str(col).zfill(5) for col in cur_patch_df.columns]
    return cur_patch_df#,cur_mask

## returns an 1D numpy array corresponding to land masks of all the relevant patches with dimension pxhxw
## p is number of patches
## h height dimension of each patch
## w width dimension of each patch
## ith index corresponds to land state of land state of x,y cell of Ath patch 
## A = i//hw, B = i%hw, x = B//w, y = B%w
def get_mask_data(processed_origins,land_mask,config):
    all_masks = []
    for patch_idx, patch_origin in enumerate(processed_origins):
        row,col = patch_origin
        cur_mask = land_mask[row:row+config.patch_size,col:col+config.patch_size]
        cur_mask = rearrange(cur_mask,'h w -> (h w)')
        all_masks.append(cur_mask)
    return rearrange(np.array(all_masks),'p hw -> (p hw)')

def get_mask_data_for_diffstg(processed_origins,land_mask,config):
    all_masks = []
    for patch_idx, patch_origin in enumerate(processed_origins):
        row,col = patch_origin
        cur_mask = land_mask[row:row+config.patch_size,col:col+config.patch_size]
        #cur_mask = rearrange(cur_mask,'h w -> (h w)')
        all_masks.append(cur_mask)
    return rearrange(np.array(all_masks),'p h w -> (p hw)')

def get_mask_data_v2(processed_origins,land_mask,config):
    all_masks = []
    for patch_idx, patch_origin in enumerate(processed_origins):
        row,col = patch_origin
        cur_mask = land_mask[row:row+config.patch_size,col:col+config.patch_size]
        cur_mask = rearrange(cur_mask,'h w -> (h w)')
        all_masks.append(cur_mask)
    return np.array(all_masks)

## directly return the nans based on inundation values
def get_mask_data_from_datamap_old(processed_origins,tidewatch_test_data,config):
    all_masks = []
    for patch_idx, patch_origin in enumerate(processed_origins):
        row,col = patch_origin
        cur_mask = tidewatch_test_data[:,row:row+config.patch_size,col:col+config.patch_size]
        #print(cur_mask[0,0:4,0:4])
        cur_mask = ~np.isnan(cur_mask)
        #print(cur_mask[0,0:4,0:4])
        all_masks.append(cur_mask)
    return np.array(all_masks)

## to get rid of vstacking
def get_mask_data_from_datamap(processed_origins,tidewatch_train_data,tidewatch_validation_data,config):
    all_masks = []
    for patch_idx, patch_origin in enumerate(processed_origins):
        row,col = patch_origin
        cur_mask = np.vstack((tidewatch_train_data[:,row:row+config.patch_size,col:col+config.patch_size],
                              tidewatch_validation_data[12:,row:row+config.patch_size,col:col+config.patch_size])) 
        #tidewatch_test_data[:,row:row+config.patch_size,col:col+config.patch_size]
        #print(cur_mask[0,0:4,0:4])
        cur_mask = ~np.isnan(cur_mask)
        #print(cur_mask[0,0:4,0:4])
        all_masks.append(cur_mask)
    return np.array(all_masks)

##repeatedly calling prepare_patch_data -- [hw1,hw2,...hwm] -- rearrange(p , hw= (p hw))) - (vw)
def prepare_elevation_data(elevation_frame,patch_idx,patch_row,patch_col,config):
    cur_elev = elevation_frame[patch_row:patch_row+config.patch_size,patch_col:patch_col+config.patch_size]
    cur_elev = rearrange(cur_elev, 'h w -> (h w)')
    return cur_elev

## creates gluonts PandasDataset for univariate processing
## returns 6 things: train_df,test_df,uni_train_ds,uni_test_ds,all_patch_df,all_elev
## train_df: a 2D array similar to one returned by prepare_patch_data_for_gluonts, but all patches are concatenated along the columns
## test_df: similar, in train_df, time goes from 0:Train_time, in test_df time goes from 0:Train_time+Test_time
## uni_train_ds: Creates a PandasDataset (gluonts) that has as many items as number of cells
### each item has the entire time series of the particular cell and the corresponding elevation
## uni test_ds: same thing, but uses test_df, number of items is same, but length of time series of each item is larger
## all_test_df: test_df deconcatenated, contains list of dataframes, each dataframe corresponds to one patch, probably needed by another method
## all_elev_data: list of elevation numpy arrays of each patch
def create_univariate_data(processed_origins,tidewatch_train_data,tidewatch_validataion_data,elevation_frame,config,tot_train_time,nan_replace=None):##

    train_dfs = []
    test_dfs = []
    all_elev_data = []
    all_masks = []
    
    for patch_idx, patch_origin in enumerate(processed_origins):
        row,col = patch_origin
        print(row,col)
        train_dfs.append(prepare_patch_data_for_gluonts(row,col,tidewatch_train_data,config,patch_idx,nan_replace=nan_replace))
        test_dfs.append(prepare_patch_data_for_gluonts(row,col,tidewatch_train_data,config,patch_idx,nan_replace=nan_replace,sample_data_valid=tidewatch_validataion_data))
        all_elev_data.append(prepare_elevation_data(elevation_frame,patch_idx,row,col,config))
        #all_masks.append(get_mask_for_patch(row,col,land_mask,config,patch_idx))
    
    
    univariate_gluont_train_df = pd.concat(train_dfs,axis=1)
    static_features = pd.DataFrame({"elev": rearrange(np.array(all_elev_data),'p c->(p c)')},
                                   index=univariate_gluont_train_df.columns.tolist())
    univariate_gluont_train_ds = PandasDataset(dict(univariate_gluont_train_df),static_features=static_features)

    print(univariate_gluont_train_df.shape)

    univariate_gluont_test_df = pd.concat(test_dfs,axis=1)
    static_features = pd.DataFrame({"elev": rearrange(np.array(all_elev_data),'p c->(p c)')},
                                   index=univariate_gluont_test_df.columns.tolist())
    univariate_gluont_test_ds = PandasDataset(dict(univariate_gluont_test_df),static_features=static_features)

    print(univariate_gluont_test_df.shape)
    #return all_masks also
    return univariate_gluont_train_df,univariate_gluont_test_df,univariate_gluont_train_ds,univariate_gluont_test_ds,test_dfs,all_elev_data

#creates gluonts ListDataset suitable for multivariate learning
## number of items in this dataset correspond to number of different patches
## target of each item is a 2D np array instead of 1D in univariate case
## target corresponds to time series of all the grids in a single patch
def create_multivariate_data_new(all_patch_df,all_elev_data, starting_hour=6, per_timestep_hr = 1):
    #multivariate_ts = univariate_df.values
    hourly_data = np.array([(starting_hour+cur_time*per_timestep_hr)%24 for cur_time in range(0,all_patch_df[0].shape[0])])
    hourly_data_for_gluonts = repeat(hourly_data,'t -> f t', f=all_patch_df[0].shape[1])
    print('feat_dynamic_real shape for one patch',hourly_data_for_gluonts.shape)
    multivariate_ds = ListDataset( [ {'start':"2024-03-28 06:00:00", 'target': (univariate_df.values).T,'patch':'patch_'+str(idx).zfill(3),
                                      'feat_static_real':elev_info, 'feat_dynamic_real': hourly_data_for_gluonts,'feat_dynamic_cat': hourly_data_for_gluonts} 
                                    for idx,(univariate_df,elev_info) in enumerate(zip(all_patch_df,all_elev_data))], 
                                  freq="H", one_dim_target=False)
    return multivariate_ds

def create_multivariate_data(all_patch_df,all_elev_data):
    #multivariate_ts = univariate_df.values
    multivariate_ds = ListDataset( [ {'start':"2024-03-28 06:00:00", 'target': (univariate_df.values).T,'patch':'patch_'+str(idx).zfill(3),
                                      'feat_static_real':elev_info } for idx,(univariate_df,elev_info) in enumerate(zip(all_patch_df,all_elev_data))] , 
                                  freq="H", one_dim_target=False)
    return multivariate_ds

#creates test instances for evaluating the model
def create_test_pairs(whole_data,prediction_length,no_windows,train_period_end):
    training_data, test_template = split(
        whole_data, date=pd.Period(train_period_end, freq="1H")
    )

    print(type(training_data),type(test_template))

    test_pairs = test_template.generate_instances(
        prediction_length=prediction_length,
        windows=no_windows,
        distance = 1
    )

    print(type(test_pairs))
    print(len(test_pairs))
    return training_data, test_pairs

## training snippet, currently supports 3 models
## training snippet, currently supports 3 models
def train(univariate_training_data,multivariate_training_data,prediction_length,context_length,config,
          model_name="SFF",MP=0.0,NOISE=True):
    print(model_name)
    if model_name=='DVAR':
        target_dim = next(iter(multivariate_training_data))['target'].shape[0]
        print('target dim is',target_dim)
        trainer = Trainer(ctx="cpu", epochs=config.num_epochs, learning_rate=config.learning_rate, num_batches_per_epoch=config.train_batch_size)
        dvar_estimator = DeepVAREstimator(
            target_dim = target_dim,
            freq='1H',
            context_length=context_length,
            prediction_length=prediction_length,
            trainer=trainer,
            num_cells=128,
            num_layers=8,
            conditioning_length=100,
            num_parallel_samples=5,
            lags_seq = [1],
            noise_percentage=MP,
            mask_data_patch_size = config.patch_size
        )
        dvar_trained_ouput = dvar_estimator.train_model(multivariate_training_data)
        trained_net = dvar_trained_ouput.trained_net
        dvar_predictor = dvar_trained_ouput.predictor
        print("model_params:",trainer.count_model_params(trained_net))
        return dvar_predictor
    
    if model_name=='GPVAR':
        target_dim = next(iter(multivariate_training_data))['target'].shape[0]
        print('target dim is',target_dim)
        trainer = Trainer(ctx="cpu", epochs=config.num_epochs, learning_rate=config.learning_rate, 
                          num_batches_per_epoch=config.train_batch_size,hybridize=False)
        gpvar_estimator = GPVAREstimator(freq = "1H",
            prediction_length = prediction_length,
            target_dim = target_dim,
            trainer = trainer,
            num_cells=32,
            num_layers=2,
            lags_seq = [1],
            num_parallel_samples = 4,
            context_length = context_length, noise_percentage=MP,
            mask_data_patch_size = config.patch_size)
        gpvar_trained_ouput = gpvar_estimator.train_model(multivariate_training_data)
        trained_net = gpvar_trained_ouput.trained_net
        gpvar_predictor = gpvar_trained_ouput.predictor
        print("model_params:",trainer.count_model_params(trained_net))
        return gpvar_predictor
    
    if model_name=='TFT':
        trainer = Trainer(ctx="cpu", epochs=config.num_epochs, learning_rate=config.learning_rate, num_batches_per_epoch=config.train_batch_size)
        tft_estimator = TemporalFusionTransformerEstimator(
            freq='1H',
            context_length=context_length,
            prediction_length=prediction_length,
            trainer=trainer,
            hidden_dim=64,
            num_heads=4
        )
        tft_trained_output = tft_estimator.train_model(univariate_training_data)
        trained_net = tft_trained_output.trained_net
        tft_predictor = tft_trained_output.predictor
        print("model_params:",trainer.count_model_params(trained_net))
        return tft_predictor
    
    if model_name=="SFF":
        trainer = Trainer(ctx="cpu", epochs=config.num_epochs, learning_rate=config.learning_rate, num_batches_per_epoch=config.train_batch_size)
        sff_estimator = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[256,128],
            prediction_length=prediction_length,##how much will they predict
            context_length=context_length,##how much previous history will see
            trainer=trainer,
        )
        sff_trained_output = sff_estimator.train_model(univariate_training_data)
        trained_net = sff_trained_output.trained_net
        sff_predictor = sff_trained_output.predictor
        print("model_params:",trainer.count_model_params(trained_net))
        return sff_predictor
    
    if model_name=="DAR":
        trainer = Trainer(epochs=config.num_epochs)
        dar_estimator = DeepAREstimator(##num_layer=2,hidden_Size=256,context_length--read from config
            freq='1H',
            context_length=context_length,
            prediction_length=prediction_length,
            trainer=trainer,
            num_cells=128,
            num_layers=4,
            use_feat_static_real=True
        )
        dar_trained_output = dar_estimator.train_model(univariate_training_data)
        trained_net = dar_trained_output.trained_net
        dar_predictor = dar_trained_output.predictor
        print("model_params:",trainer.count_model_params(trained_net))
        return dar_predictor
    
    if model_name=="LSTNET":
        target_dim = next(iter(multivariate_training_data))['target'].shape[0]
        trainer = Trainer(model_name=model_name,ctx='cpu', epochs=config.num_epochs, hybridize=True,
                  num_batches_per_epoch=config.train_batch_size, 
                  learning_rate=config.learning_rate, weight_decay=1e-4,mask_percentage=MP,noise_mask=NOISE)
        
        lstnet_estimator = LSTNetEstimator(
            skip_size=2, ar_window=4, num_series=target_dim,
            rnn_num_layers=2, skip_rnn_num_layers=1,
            channels=4, kernel_size=4, dropout_rate=0.2, output_activation='sigmoid',
            prediction_length=prediction_length, context_length=context_length, noise_percentage=MP,
            mask_data_patch_size = config.patch_size,
            trainer=trainer
        )
        #lstnet_predictor=lstnet_estimator.train(multivariate_training_data)
        lstnet_trained_output=lstnet_estimator.train_model(multivariate_training_data)
        trained_net = lstnet_trained_output.trained_net
        print("model_params:",trainer.count_model_params(trained_net))
        lstnet_predictor = lstnet_trained_output.predictor
        ## pass a trained predictor for next training batch
        ## each patch 
        return lstnet_predictor
    if model_name=='TIMEGRAD':
        
        target_dim = next(iter(multivariate_training_data))['target'].shape[0]
        scheduler = DEISMultistepScheduler(
            num_train_timesteps=150,
            beta_end=0.1,
        )
        timegrad_estimator = TimeGradEstimator(
            input_size=target_dim,
            hidden_size=32,
            num_layers=4,
            dropout_rate=0.1,
            lags_seq=[1],
            scheduler=scheduler,
            num_inference_steps=149,
            prediction_length=prediction_length,
            context_length=context_length,
            freq='H',
            scaling=True,
            imputation_method=None,
            noise_percentage=MP,
            mask_data_patch_size = config.patch_size,
            trainer_kwargs=dict(max_epochs=config.num_epochs, accelerator="gpu", devices="1",
                                default_root_dir='/project/biocomplexity/Diff_Spatio_Temporal_Grid/Models/Timegrad/'),
        )
        
        timegrad_predictor = timegrad_estimator.train(multivariate_training_data, cache_data=True, shuffle_buffer_length=64)
        return timegrad_predictor

## training snippet, currently supports 3 models
#from gluonts.torch import SimpleFeedForwardEstimator, DeepAREstimator

    
## use test instances to evaluate a predictor model
def get_forecasts(test_pair,predictor,num_samples):
    forecasts = list(predictor.predict(test_pair.input,num_samples=num_samples))
    all_forecasts = np.array([forecasts[i].samples for i in range(len(forecasts))])
    return all_forecasts


## use forecast and ground truth of test instances to get crps score of a model
## use forecast and ground truth of test instances to get crps score of a model
def get_crps(all_forecasts,test_pairs,gluonts_mask_data,model_type='univariate'): #all_masks,#config
    #vector (phw)
    if model_type=='univariate':
        all_forecasts = rearrange(all_forecasts,'vw s h -> vw h s') ##
        all_forecasts_land_only = all_forecasts[gluonts_mask_data,:,:]
        
        all_truths = np.array([test_pair[1]['target'] for test_pair in list(test_pairs)]) #shape is 'v h' ## (vw, h)
        all_truths_land_only = all_truths[gluonts_mask_data,:]
    
    else:   
        all_forecasts = rearrange(all_forecasts,'w s h v-> (w v) h s')
        all_forecasts_land_only = all_forecasts[gluonts_mask_data,:,:]
        #all_forecasts_land_only = all_forecasts*all_masks_forecast
        
        all_truths = np.array([test_pair[1]['target'] for test_pair in list(test_pairs)])
        all_truths = rearrange(all_truths,'w v h-> (w v) h')
        all_truths_land_only = all_truths[gluonts_mask_data,:]
    
    print(all_truths.shape,all_truths_land_only.shape)
    print(all_forecasts.shape,all_forecasts_land_only.shape)
    
    crps = ps.crps_ensemble(all_truths, all_forecasts)
    crps_land_only = ps.crps_ensemble(all_truths_land_only, all_forecasts_land_only)
    
    val_loss = crps.sum()/np.sum(np.absolute(all_truths)) # https://arxiv.org/pdf/2401.03006
    val_loss_land_only = crps_land_only.sum()/np.sum(np.absolute(all_truths_land_only))
    
    all_forecasts_average = np.mean(all_forecasts,axis=-1)
    rmse = (np.mean((all_forecasts_average-all_truths)*(all_forecasts_average-all_truths)))**0.5
    rmse = rmse/(np.max(all_truths)-np.min(all_truths))
    
    all_forecasts_average_land_only = np.mean(all_forecasts_land_only,axis=-1)
    rmse_land_only = (np.mean((all_forecasts_average_land_only-all_truths_land_only)*(all_forecasts_average_land_only-all_truths_land_only)))**0.5
    rmse_land_only = rmse_land_only/(np.max(all_truths_land_only)-np.min(all_truths_land_only))
    
    return val_loss,rmse,val_loss_land_only,rmse_land_only#return land_only_losses


def get_crps_naive(all_forecasts,test_pairs,gluonts_mask_data,model_type='univariate'): #all_masks,#config
    #vector (phw)
    gluonts_mask_data_naive = gluonts_mask_data[0:32*32]
    gluonts_mask_data_naive = gluonts_mask_data_naive.astype(bool)
    #all_forecasts = all_forecasts[0:1,:,:,:]
    if model_type=='univariate':
        all_forecasts = rearrange(all_forecasts,'vw s h -> vw h s') ##
        all_forecasts_land_only = all_forecasts[gluonts_mask_data,:,:]
        
        all_truths = np.array([test_pair[1]['target'] for test_pair in list(test_pairs)]) #shape is 'v h' ## (vw, h)
        all_truths_land_only = all_truths[gluonts_mask_data,:]
    
    else:   
        all_forecasts = rearrange(all_forecasts,'w s h v-> (w v) h s')
        all_forecasts_land_only = all_forecasts[gluonts_mask_data,:,:]
        #all_forecasts_land_only = all_forecasts[gluonts_mask_data_naive,:,:]
        #all_forecasts_land_only = all_forecasts*all_masks_forecast
        
        all_truths = np.array([test_pair[1]['target'] for test_pair in list(test_pairs)])
        #all_truths = np.array([test_pair[1]['target'] for test_pair in list(test_pairs)[0:1]])
        all_truths = rearrange(all_truths,'w v h-> (w v) h')
        all_truths_land_only = all_truths[gluonts_mask_data,:]
        #all_truths_land_only = all_truths[gluonts_mask_data_naive,:]
        print(all_truths_land_only.shape)
        print('nan in all_truth_land',np.isnan(all_truths_land_only).any())
    
    print(all_truths.shape,all_truths_land_only.shape)
    print(all_forecasts.shape,all_forecasts_land_only.shape)
    
    crps = ps.crps_ensemble(all_truths, all_forecasts)
    crps_land_only = ps.crps_ensemble(all_truths_land_only, all_forecasts_land_only)
    
    val_loss = crps.sum()/np.sum(np.absolute(all_truths)) # https://arxiv.org/pdf/2401.03006
    val_loss_land_only = crps_land_only.sum()/np.sum(np.absolute(all_truths_land_only))
    
    all_forecasts_average = np.mean(all_forecasts,axis=-1)
    rmse = (np.mean((all_forecasts_average-all_truths)*(all_forecasts_average-all_truths)))**0.5
    rmse = rmse/(np.max(all_truths)-np.min(all_truths))
    
    all_forecasts_average_land_only = np.mean(all_forecasts_land_only,axis=-1)
    rmse_land_only = (np.mean((all_forecasts_average_land_only-all_truths_land_only)*(all_forecasts_average_land_only-all_truths_land_only)))**0.5
    rmse_land_only = rmse_land_only/(np.max(all_truths_land_only)-np.min(all_truths_land_only))
    
    return val_loss,rmse,val_loss_land_only,rmse_land_only#return land_only_losses

def get_crps_v3(all_truth,all_forecast,all_mask,model_type='univariate'): #all_masks,#config
    # all_truth # tp,v,h
    # all_mask # tp,v,h
    # all_forecast #tp,s,v,h
    
    all_truths = rearrange(all_truth,'tp v h -> (tp v h)')
    all_mask = rearrange(all_mask,'tp v h -> (tp v h)')
    all_forecasts = rearrange(all_forecast,'tp s v h -> (tp v h) s')
    
    all_truths_land_only = all_truths[all_mask]
    all_forecasts_land_only = all_forecasts[all_mask,:]
    
    crps = ps.crps_ensemble(all_truths, all_forecasts)
    crps_land_only = ps.crps_ensemble(all_truths_land_only, all_forecasts_land_only)
    
    nacrps = crps.sum()/np.sum(np.absolute(all_truths)) # https://arxiv.org/pdf/2401.03006
    nacrps_land_only = crps_land_only.sum()/np.sum(np.absolute(all_truths_land_only))
    
    mean_forecasts = reduce(all_forecasts,'v s -> v','mean')
    #all_forecasts_average = np.mean(all_forecasts,axis=-1)
    rmse = np.sqrt(np.mean(np.square(mean_forecasts-all_truths)))
    nrmse = rmse/(np.max(all_truths)-np.min(all_truths))
    
    mean_forecasts_land_only = np.mean(all_forecasts_land_only,axis=-1)
    rmse_land_only = np.sqrt(np.mean(np.square(mean_forecasts_land_only-all_truths_land_only)))
    nrmse_land_only = rmse_land_only/(np.max(all_truths_land_only)-np.min(all_truths_land_only))
    
    return nacrps,nrmse,nacrps_land_only,nrmse_land_only#return land_only_losses
