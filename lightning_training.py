import torch
import random
import numpy as np
import torch.nn as nn

import json
from dataclasses import asdict as dataclass_to_dict

from torch.utils.data import DataLoader

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from utils import check_gpu_status
from training_config import TrainingConfig

from TidewatchDataset import TidewatchTimeseriesDataset
from TidewatchPatchDataset import MultiPatchDataset

from diffusion import SpatioTemporalDDPM
from diffusers import UNet2DConditionModel
from hidden_state_net import HiddenStateNet
from patch_embedding import PatchEmbeddingNet
from lightning_diffusion import LightningDiffusionModel

import argparse
import pickle

from torch.utils.data import Dataset, Subset


def get_train_dataset(config, patch_origins):
    train_dataset = TidewatchTimeseriesDataset(
        root_dir='data/Tiff/',
        start_dt=config.train_start_datetime,
        end_dt=config.train_end_datetime,
        context_length=config.context_length,
        horizon_length=config.training_horizon_length,
        elevation_file='ESVA_Elevation_EPSG_3857_USGS30m.tif',
        covariate_file='exmore_VA_covariates_2024-03-28_to_2024-07-01.csv'
    )

    multi_patch_train_dataset = MultiPatchDataset(
        train_dataset,
        patch_origins=patch_origins,
        config=config,
        shuffle_patch=config.shuffle_patch
    )

    print("# of Unique Patches Train Dataset:", multi_patch_train_dataset.number_of_patches)
    print("Train Dataset length:", len(multi_patch_train_dataset))
    return train_dataset, multi_patch_train_dataset


def get_validation_dataset(config, patch_origins, multi_patch_train_dataset):
    validation_dataset = TidewatchTimeseriesDataset(
        root_dir='data/Tiff/',
        start_dt=config.validation_start_datetime,
        end_dt=config.validation_end_datetime,
        context_length=config.context_length,
        horizon_length=config.validation_horizon_length,
        elevation_file='ESVA_Elevation_EPSG_3857_USGS30m.tif',
        covariate_file='exmore_VA_covariates_2024-03-28_to_2024-07-01.csv'
    )

    multi_patch_validation_dataset = MultiPatchDataset(
        validation_dataset,
        patch_origins=patch_origins,
        config=config,
        shuffle_patch=False,
        mean_inundation=multi_patch_train_dataset.mean_inundation,
        std_inundation=multi_patch_train_dataset.std_inundation,
        mean_elevation=multi_patch_train_dataset.mean_elevation,
        std_elevation=multi_patch_train_dataset.std_elevation
    )

    print("# of Unique Patches Validation Dataset:", multi_patch_validation_dataset.number_of_patches)
    print("Validation Dataset length:", len(multi_patch_validation_dataset))
    return multi_patch_validation_dataset


def get_test_dataset(config, patch_origins, test_masks, multi_patch_train_dataset):
    test_dataset = TidewatchTimeseriesDataset(
        root_dir='data/Tiff/',
        start_dt=config.test_start_datetime,
        end_dt=config.test_end_datetime,
        context_length=config.context_length,
        horizon_length=config.test_horizon_length,
        elevation_file='ESVA_Elevation_EPSG_3857_USGS30m.tif',
        covariate_file='exmore_VA_covariates_2024-03-28_to_2024-07-01.csv'
    )

    multi_patch_test_dataset = MultiPatchDataset(
        test_dataset,
        patch_origins=patch_origins,
        config=config,
        masks=test_masks,
        shuffle_patch = False,
        mean_inundation=multi_patch_train_dataset.mean_inundation,
        std_inundation=multi_patch_train_dataset.std_inundation,
        mean_elevation=multi_patch_train_dataset.mean_elevation,
        std_elevation=multi_patch_train_dataset.std_elevation
    )
    print("# of Unique Patches Test Dataset:", multi_patch_test_dataset.number_of_patches)
    print("Test Dataset length:", len(multi_patch_test_dataset))
    return multi_patch_test_dataset


def get_conditional_unet(config):
    unet2d_model = UNet2DConditionModel(
        sample_size=(config.patch_size, config.patch_size),  # the target image resolution
        in_channels=1,  # the number of input channels
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block,
        norm_num_groups=16,
        block_out_channels=(16, 32, 32, 64),
        cross_attention_dim=config.spatial_embedding_size,

        # class_embed_type='simple_projection',
        # projection_class_embeddings_input_dim=config.spatial_embedding_size,
        # class_embeddings_concat=False,

        # only_cross_attention=True,
        dropout=config.dropout,

        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),

        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
    )

    return unet2d_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help='Seed for random generator.')
    parser.add_argument("--storePath", type=str, required=True, help='Store path for models.')
    args = parser.parse_args()

    # Chck for GPU
    check_gpu_status()

    # Setting reproducibility
    SEED = int(args.seed)
    print(f"Random Seed: {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Training configurations
    config = TrainingConfig()
    config.store_path = args.storePath

    patch_origins = [ (int (x), int(y)) for x, y, _ in pickle.load(open(config.patch_origins, 'rb'))]
    print(f"Patch Origins: {patch_origins[:config.num_unique_patches]}")

    test_masks = torch.load(config.test_masks)
    print(f"test masks shape: {test_masks.shape}")

    # Get train dataset and define train dataloader
    tidewatch_train_dataset, multi_patch_train_dataset = get_train_dataset(config, patch_origins)
    train_dataloader = DataLoader(multi_patch_train_dataset, batch_size=config.train_batch_size, shuffle=False, num_workers=8)

    config.train_mean_inundation = multi_patch_train_dataset.mean_inundation
    config.train_std_inundation = multi_patch_train_dataset.std_inundation
    config.train_mean_elevation = multi_patch_train_dataset.mean_elevation
    config.train_std_elevation = multi_patch_train_dataset.std_elevation
    json.dump(dataclass_to_dict(config), open(f"{config.store_path}/config.json", 'w'), default=str)
    # json.dump(dataclass_to_dict(config), open(f"{config.store_path}/config_test_{config.test_horizon_length}.json", 'w'), default=str)

    multi_patch_validation_dataset = get_validation_dataset(config, patch_origins, multi_patch_train_dataset)
    val_dataloader = DataLoader(multi_patch_validation_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=8)

    multi_patch_test_dataset = get_test_dataset(config, patch_origins, test_masks, multi_patch_train_dataset)
    test_dataloader = DataLoader(multi_patch_test_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=8)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_masked_nrmse',
        mode='min',
        save_top_k=config.save_top_k,
        save_last=True,
        dirpath=config.store_path,  # Directory where the checkpoints will be saved
        filename='{epoch}-{val_masked_nrmse:.6f}'  # Checkpoint file naming pattern
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Define the Trainer
    trainer = Trainer(
        accelerator="cuda",
        precision="32-true",
        max_epochs=config.num_epochs,
        limit_val_batches=config.num_val_batches,
        check_val_every_n_epoch=1,
        # limit_train_batches=10,
        devices=1,
        default_root_dir=config.store_path,
        # log_every_n_steps=50,
        # enable_checkpointing=False
        # strategy='deepspeed_stage_2',
        callbacks=[checkpoint_callback, lr_monitor]
    )
    
    with trainer.init_module():
        # models created here will be on GPU
        # Get UNet and HiddenStateNet
        unet2d_model = get_conditional_unet(config)
        hidden_state_model = HiddenStateNet(config=config)
        patch_embedding_model = PatchEmbeddingNet(config=config) if config.use_patch_embedding else None
    
        diffusion_model = SpatioTemporalDDPM(
            unet2d_model,
            hidden_state_model,
            patch_embedding_model,
            config=config
        )
    
        # Define the lightning model. Compile for faster training.
        model = LightningDiffusionModel(diffusion_model=diffusion_model, config=config)
        # print("Compiling the model ...")
        # compiled_model = torch.compile(model)
        # print("Compiling complete!")

    # Run training
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    ###############  Testing  ###################
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    trainer.test(
        # model,
        # ckpt_path=f"{config.store_path}/best.ckpt",
        ckpt_path="best",
        dataloaders=test_dataloader
    )

    # print(f"Best model path: {config.store_path}/best.ckpt")
    # trainer.test(
    #     model,
    #     ckpt_path=f"{config.store_path}/best.ckpt",
    #     dataloaders=test_dataloader
    # )
    