# DIFF-SPARSE: Towards High Resolution Probabilistic Coastal Inundation Forecasting from Sparse Observations

[![arXiv](https://img.shields.io/badge/arXiv-2505.05381-b31b1b.svg)](https://arxiv.org/abs/2505.05381)
[![AAAI 2026](https://img.shields.io/badge/AAAI%202026-AI%20for%20Social%20Impact-blue.svg)](https://aaai.org/conference/aaai/aaai-26/aisi-call/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Official implementation of **DIFF-SPARSE**, a masked conditional diffusion model for probabilistic coastal inundation forecasting from sparse sensor observations.

**Authors:** Kazi Ashik Islam, Zakaria Mehrab, Mahantesh Halappanavar, Henning Mortveit, Sridhar Katragadda, Jon Derek Loftis, Stefan Hoops, Madhav Marathe

**Accepted at:** AAAI 2026 Conference on Artificial Intelligence (AI for Social Impact Track)

---

## Overview

Coastal flooding poses increasing threats to communities worldwide, yet sensor networks for monitoring inundation are often sparse and costly to deploy. DIFF-SPARSE addresses this challenge by enabling **high-resolution probabilistic forecasting** of coastal inundation from sparse observations.

### Key Features

- **Sparse Data Handling**: Trained with a novel masking strategy to handle up to 95% missing observations
- **Probabilistic Forecasting**: Generates multiple scenarios for uncertainty quantification
- **Physics-Informed**: Model is trained on physics-based hydrodynamic simulation data.
- **Spatio-Temporal Context**: Leverages historical data from target and neighboring locations
- **High Performance**: Up to **62% improvement** over existing methods at 95% sparsity

---

## Installation

### Requirements

- Python 3.12+
- CUDA-capable GPU (recommended)
- 64GB RAM (for full-resolution processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/KAI10/Diff-Sparse.git
cd Diff-Sparse
```
Create conda environment using [conda_environment.yml](conda_environment.yml). Use appropriate `<env-name>` and `<user-name>`.

---

## Dataset

### Tidewatch Virginia Dataset

Our experiments use coastal inundation data from **Virginia's Eastern Shore** (March-July 2024).

**Download:** [Net.Science Data Repository](https://net.science/files/79077d75-7dde-4c1b-964b-4f6a828dcaf4)

**Dataset Structure:**
```
data/
├── Tiff/                          # Hourly inundation rasters (3.9GB)
│   ├── MM_DD_YYYY_HH_MM_SS.tif    # Hourly snapshots
│   ├── ESVA_Elevation_EPSG_3857_USGS30m.tif
│   └── exmore_VA_covariates_2024-03-28_to_2024-07-01.csv
├── Patches/                       # Patch coordinate definitions
│   └── non_overlapping_patches_{16,32,64,80,96}.pickle
└── Test-Masks/                    # Sparsity masks for evaluation
    └── masks_{patch_size}_{sparsity}_{seed}.pt
```

**Setup:**
1. Download the dataset from the link above.
2. Extract to `data/` directory.
3. Ensure the directory structure matches the layout above.

---

## Quick Start

### Configuration

Edit `training_config.py` to set patch configurations and hyperparameters:

```python
@dataclass
class TrainingConfig:
    patch_size: int                     # Patch dimensions
    train_batch_size: int               # Training batch size
    eval_batch_size: int                # Evaluation batch size
    num_unique_patches: int             # Number of spatial patches
    patch_origins: str                  # File containing patch origins

    train_start_datetime: datetime      # Train start datetime
    train_end_datetime: datetime        # Train end datetime
    test_start_datetime: datetime       # test start datetime 
    test_end_datetime: datetime         # test end datetime 
    validation_start_datetime: datetime # validation start datetime
    validation_end_datetime: datetime   # validation end datetime

    num_val_batches: int                # Number of batches for validation
    num_scenarios_validation: int       # Number of scenarios to predict during validation
    num_scenarios_test: int             # Number of scenarios to predict during test
    shuffle_patch: bool                 # Shuffle patch in MultiPatchDataset
    
    context_length: int                 # Historical context length (hours)
    training_horizon_length: int        # Horizon length during training (must be 1).
    test_horizon_length: int            # Horizon length during test.
    validation_horizon_length: int      # Horizon length during validation (should be same as test_horizon_length).

    num_channels: int                   # Number of input channels (inundation, data/sensor_mask, elevation)
    add_data_mask: bool                 # Add sensor_mask as separate channel
    add_elevation_channel: bool         # Add elevation matrix as separate channel.
    use_covariate_embedding: bool       # Use temporal-covariate embedding

    spatial_embedding_size: int         # Dimension of context embedding after convolution
    linear_layer_size: int              # Dimension of final context embedding
    dropout: float                      # Dropout in UNet2DConditional
    covariate_dimension: int            # Dimension of temporal-covariate embedding

    data_missing_percentage: float      # Percentage of missing sensors / sparsity level

    num_test_masks: int                 # Number of test masks
    test_masks: str                     # File containing test masks

    num_diffusion_steps: int            # Number of DDPM denoising steps
    min_beta: float                     # DDPM hyper-parameter
    max_beta: float                     # DDPM hyper-parameter

    save_top_k: int                     # Number of best models to save during training
    store_path: str                     # Path to store best models

    train_mean_inundation: float        # Will be set automatically during training
    train_std_inundation: float         # Will be set automatically during training
    train_mean_elevation: float         # Will be set automatically during training
    train_std_elevation: float          # Will be set automatically during training

    use_patch_embedding: bool           # Not used.
    consistency_loss_weight: float      # Not used.
```

---

### Training

```bash
# Train with configurations set in training_config.py
python lightning_training.py \
    --seed=0 \
    --storePath=output
```

### Batch Training on HPC Cluster

```bash
# Submit 10 jobs with different random seeds via SLURM
bash driver.sh

# Customize configuration in driver.sh:
# - patchSize: {16, 32, 64, 80, 96}; should be same as patch_size
# - numPatches: Number of unique patches; should be same as num_unique_patches 
# - ECH: (1) Use elevation data, (0) Do not use elevation data. Should be same as add_elevation_channel.
# - COV: (1) Use temporal covariates, (0) Do not use temporal covariates. Should be same as use_covariate_embedding.
# - MASKP: Sparsity level; should be same as data_missing_percentage
```

### Testing

Testing is done on test data defined in [training_config.py](training_config.py) right after training.

### Generate Test Masks

```bash
# Create custom sparsity masks
python generate_test_masks.py
```

---

## Baseline Comparisons

See [baseline-comparison-instructions](baseline-scripts/README.MD).

---

## Project Structure

```
Diff-Sparse/
├── diffusion.py                    # DDPM implementation
├── lightning_diffusion.py          # PyTorch Lightning module
├── lightning_training.py           # Main training script
├── hidden_state_net.py             # 3D CNN encoder
├── training_config.py              # Patch configuration & Hyperparameters 
├── TidewatchDataset.py             # Tidewatch dataset class
├── TidewatchPatchDataset.py        # Multi-patch dataset class
├── datastore.py                    # Memory-mapped data access
├── generate_test_masks.py          # Sparse sensor mask generation for testing
├── utils.py                        # Utility functions
├── driver.sh                       # SLURM batch submission
├── worker.sh                       # SLURM worker script
├── baseline-scripts/               # Baseline method scripts
│   ├── DCRNN/
│   ├── DiffSTG/
│   ├── bayesnf-gluonts/
│   └── gluonts/
├── data/                           # Data directory (download separately)
└── conda_environment.yml           # Conda environment
```

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{islam2026diffsparse,
  doi = {10.48550/ARXIV.2505.05381},
  url = {https://arxiv.org/abs/2505.05381},
  author = {Islam,  Kazi Ashik and Mehrab,  Zakaria and Halappanavar,  Mahantesh and Mortveit,  Henning and Katragadda,  Sridhar and Loftis,  Jon Derek and Hoops,  Stefan and Marathe,  Madhav},
  keywords = {Machine Learning (cs.LG),  FOS: Computer and information sciences},
  title = {Towards High Resolution Probabilistic Coastal Inundation Forecasting from Sparse Observations},
  publisher = {arXiv},
  year = {2025},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

_Updated citation will be posted once the AAAI 2026 proceedings are published._

**Preprint:** [https://arxiv.org/abs/2505.05381](https://arxiv.org/abs/2505.05381)

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

See [LICENSE](https://creativecommons.org/licenses/by/4.0/) for details.

---
