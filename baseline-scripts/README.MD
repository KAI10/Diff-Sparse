In order to make fair comparison, we made changes to the gluonts package so that they can work with dynamic sensor mask. Please replace the `gluonts` directory with the one given here to have those changes reflected

To run BayesNF:

`python coastal_bayesnf.py <patch_size> <num_patches> <SEED>`

Example: `python coastal_bayesnf.py 16 2 0`

To run DiffSTG:

`python train_tidewatch_diffstg.py --seed <SEED> --data <patch_conf> --noise_mask <noise>`

Example: `python train_tidewatch_diffstg.py --seed 0 --data TIDWATCH_16_2 --noise_mask 0.95`

To run DCRNN:

`python train.py --config config_<data_name>.json`

Example: `python train.py --config config_tidewatch_16_2.json`

To run LSTNET/DEEPVAR/TIMEGRAD:

`python baseline_inundation_with_noise.py <SEED> <patch_size> <num_patches> <model_name>`

Example: `python baseline_inundation_with_noise.py 0 16 2 LSTNET`

Any data will be provided upon requests