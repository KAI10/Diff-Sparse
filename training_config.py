from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrainingConfig:
    patch_size: int = 64
    train_batch_size: int = 32
    eval_batch_size: int = 4
    num_unique_patches: int = 10

    patch_origins: str = f'data/Patches/non_overlapping_patches_{patch_size}.pickle'

    train_start_datetime: datetime = datetime.strptime('03-28-2024-06AM', '%m-%d-%Y-%I%p')
    train_end_datetime: datetime = datetime.strptime('06-17-2024-05PM', '%m-%d-%Y-%I%p')

    test_start_datetime: datetime = datetime.strptime('06-17-2024-06AM', '%m-%d-%Y-%I%p')
    test_end_datetime: datetime = datetime.strptime('06-24-2024-05PM', '%m-%d-%Y-%I%p')

    validation_start_datetime: datetime = datetime.strptime('06-24-2024-06AM', '%m-%d-%Y-%I%p')
    validation_end_datetime: datetime = datetime.strptime('07-01-2024-05PM', '%m-%d-%Y-%I%p')

    # 1 Day
    # Number of frames: 24 + 12 = 36
    # Number of samples: 36 - (12 + 12) + 1 = 13
    # Number of samples for 1 patch: 13, 2 patches: 13 * 2 = 26, 5 patches: 13 * 5 = 65, 10 patches: 13 * 10 = 130, 20 patches: 13 * 20 = 260
    # Number of batches for 1 patch: 13/4 = 3.25, 2 patches: 26 / 4 = 6.5, 5 patches: 65 / 4 = 16.25, 10 patches: 130 / 4 = 32.5, 20 patches: 260 / 4 = 65

    # 2 Days
    # Number of frames: 2*24 + 12 = 60
    # Number of samples: 60 - (12 + 12) + 1 = 37
    # Number of samples for 1 patch: 37, 2 patches: 37 * 2 = 74, 5 patches: 37 * 5 = 185, 10 patches: 37 * 10 = 370, 20 patches: 37 * 20 = 740
    # Number of batches for 1 patch: 37/4 = 9.25, 2 patches: 74 / 4 = 18.5, 5 patches: 185 / 4 = 46.25, 10 patches: 370 / 4 = 92.5, 20 patches: 740 / 4 = 185

    # 7 Days
    # Number of frames: 7*24 + 12 = 180
    # Number of samples: 180 - (12 + 12) + 1 = 157
    # Number of samples for 1 patch: 157, 2 patches: 157 * 2 = 314, 5 patches: 157 * 5 = 785, 10 patches: 157 * 10 = 1570, 20 patches: 157 * 20 = 3140
    # Number of batches for 1 patch: 157 / 4 = 39.25, 2 patches: 314 / 4 = 78.5, 5 patches: 785 / 4 = 196.25, 10 patches: 1570 / 4 = 392.5, 20 patches: 3140 / 4 = 785

    # Validate using 1 or 2 or 4 Day(s) of Data
    num_val_batches: int = 33

    num_scenarios_validation: int = 2
    num_scenarios_test: int = 8

    shuffle_patch: bool = True

    num_epochs: int = 40
    learning_rate: float = 1e-3
    lr_update_step_size: int = 5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 3
    
    context_length: int = 12 # 4, 8, 12, 16, 20

    training_horizon_length: int = 1
    validation_horizon_length: int = 12
    test_horizon_length: int = 12  # 1, 4, 12, 20, 28, 36

    num_channels: int = 3
    add_data_mask_channel: bool = True
    add_elevation_channel: bool = True
    use_covariate_embedding: bool = True
    use_patch_embedding: bool = False
    
    spatial_embedding_size: int = 32  # 16: 32, 32: 32, 64: 32, 80: 64, 96: 96, 128: 128, 228: 128
    linear_layer_size: int = 16 # 16: 36, 32: 25, 64: 16, 80: 36, 96: 64, 128: 144, 228: 625
    dropout: float = 0 # 0.25
    
    covariate_dimension: int = 4

    consistency_loss_weight: float = 0  # 0.25
    data_missing_percentage: float = 0.95

    num_test_masks: int = 10
    test_masks: str = f'data/Test-Masks/masks_{patch_size}_{data_missing_percentage}_{num_test_masks}.pt'
    
    # num_diffusion_steps: int = 1000
    # min_beta: float = 10 ** -4
    # max_beta: float = 0.02

    # num_diffusion_steps: int = 50
    # min_beta: float = 10 ** -4
    # max_beta: float = 0.4

    num_diffusion_steps: int = 20
    min_beta: float = 10 ** -4
    max_beta: float = 1

    # num_diffusion_steps: int = 100
    # min_beta: float = 10 ** -4
    # max_beta: float = 0.2

    # num_diffusion_steps: int = 250
    # min_beta: float = 10 ** -4
    # max_beta: float = 0.08

    # num_diffusion_steps: int = 500
    # min_beta: float = 10 ** -4
    # max_beta: float = 0.04
    
    save_top_k: int = 2
    store_path: str = ""

    train_mean_inundation: float = 0
    train_std_inundation: float = 0

    train_mean_elevation: float = 0
    train_std_elevation: float = 0
