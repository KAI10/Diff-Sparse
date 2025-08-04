import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from einops import rearrange, repeat, reduce
from lightning import LightningModule

import properscoring as ps

from consistency_loss import ConsistencyLoss

from diffusion import SpatioTemporalDDPM
from training_config import TrainingConfig


class LightningDiffusionModel(LightningModule):
    def __init__(self, diffusion_model: SpatioTemporalDDPM, config: TrainingConfig):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.config = config

        self.consistency_loss = ConsistencyLoss()
        self.consistency_loss_weight = self.config.consistency_loss_weight

        # For Validation Metric
        self.val_masked_crps = 0
        self.val_masked_gt_abs_sum = 0

        self.val_masked_squared_error = 0
        self.val_masked_gt_num = 0
        self.val_masked_gt_min, self.val_masked_gt_max = 1e9, -1e9

        # For Test Metric
        self.test_masked_crps = 0
        self.test_masked_gt_abs_sum = 0

        self.test_masked_squared_error = 0
        self.test_masked_gt_num = 0
        self.test_masked_gt_min, self.test_masked_gt_max = 1e9, -1e9

    def training_step(self, batch, batch_idx):
        context, covariate, horizon, horizon_mask = batch
        # print(f"covariate.shape: {covariate.shape}")
        covariate = covariate[:, :-self.config.training_horizon_length, :]  # Removing covariates at horizon timesteps
        # print(f"covariate.shape: {covariate.shape}")

        x0 = horizon
        x0 = torch.squeeze(x0, dim=1)  # Assuming training_horizon_length is 1
        n = len(x0)

        # print(x0.shape)
        # print(context.shape, covariate.shape, horizon.shape)

        # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
        eta = torch.randn_like(x0, device=self.device)
        t = torch.randint(0, self.config.num_diffusion_steps, (n,), device=self.device)

        # Computing the noisy image based on x0 and the time-step (forward process)
        noisy_imgs = self.diffusion_model(x0, t, eta)
        # print("noisy_imgs:", noisy_imgs)

        # Getting model estimation of original images
        condition_embedding = self.diffusion_model.get_condition_embedding(context, covariate)
        if self.config.use_patch_embedding:
            class_embedding = self.diffusion_model.get_class_embedding(
                context[:, 0, 1, :, :]  # Grabbing elevation from context
            )
        else:
            class_embedding = None

        x0_theta = self.diffusion_model.backward(
            noisy_imgs, condition_embedding, class_embedding, t.reshape(n, )
        )

        # predicted_initial_image = self.diffusion_model.denoise(noisy_imgs, eta_theta, t)
        # elevation = context[:, 0, 1:2, :, :]  # Grabbing elevation from context
        # print('elevation.shape:', elevation.shape)

        # eta_loss = F.mse_loss(eta_theta, eta)
        # denormalized_x0 = x0 * self.config.train_std_inundation + self.config.train_mean_inundation
        # denormalized_x0_theta = x0_theta * self.config.train_std_inundation + self.config.train_mean_inundation
        # denormalized_elevation = elevation * self.config.train_std_elevation + self.config.train_mean_elevation
        # c_loss = self.consistency_loss.get_loss(denormalized_x0_theta, denormalized_elevation, horizon_mask.float())
        # original_closs = self.consistency_loss.get_loss(denormalized_x0, denormalized_elevation, horizon_mask.float())
        # print(f"Consistency loss: {c_loss}, Original consistency loss: {original_closs}")

        # mse_loss = F.mse_loss(x0_theta * land_mask, x0 * land_mask)
        mse_loss = F.mse_loss(x0_theta, x0)
        train_loss = mse_loss
        # train_loss = mse_loss + self.consistency_loss_weight * c_loss
        # train_loss = (1 - self.consistency_loss_weight) * mse_loss + self.consistency_loss_weight * c_loss
        # train_loss = 1 * mse_loss

        # self.log("mse_loss", mse_loss, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("c_loss", c_loss, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("original_closs", original_closs, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def configure_optimizers(self):
        optimizer = Adam(self.diffusion_model.parameters(), lr=self.config.learning_rate)
        # lr_scheduler = StepLR(
        #     optimizer, 
        #     step_size=self.config.lr_update_step_size,
        #     gamma=self.config.lr_scheduler_factor
        # )
        # return [optimizer], [lr_scheduler]
        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_masked_nrmse",
                "frequency": 1
            }
        }

    def validation_step(self, batch, batch_idx):
        batch_masked_crps, batch_masked_gt_abs_sum, batch_masked_se, batch_masked_gt_min, batch_masked_gt_max, batch_masked_gt_num = self.process_validation_test_batch(
            batch, batch_idx, mode='validation'
        )
        self.val_masked_crps += batch_masked_crps
        self.val_masked_gt_abs_sum += batch_masked_gt_abs_sum

        self.val_masked_squared_error += batch_masked_se
        self.val_masked_gt_min = min(self.val_masked_gt_min, batch_masked_gt_min)
        self.val_masked_gt_max = max(self.val_masked_gt_max, batch_masked_gt_max)
        self.val_masked_gt_num += batch_masked_gt_num

    def on_validation_epoch_end(self):
        # nacrps = self.val_crps / self.val_gt_abs_sum
        masked_nacrps = self.val_masked_crps / self.val_masked_gt_abs_sum

        # print(f"self.val_squared_error: {self.val_squared_error}")
        # print(f"self.val_gt_num: {self.val_gt_num}")
        # print(f"self.val_gt_max: {self.val_gt_max}")
        # print(f"self.val_gt_min: {self.val_gt_min}")
        # nrmse = torch.sqrt(self.val_squared_error / self.val_gt_num) / (self.val_gt_max - self.val_gt_min)

        # print(f"self.val_masked_squared_error: {self.val_masked_squared_error}")
        # print(f"self.val_masked_gt_num: {self.val_masked_gt_num}")
        # print(f"self.val_masked_gt_max: {self.val_masked_gt_max}")
        # print(f"self.val_masked_gt_min: {self.val_masked_gt_min}")
        masked_nrmse = torch.sqrt(self.val_masked_squared_error / self.val_masked_gt_num) / (self.val_masked_gt_max - self.val_masked_gt_min)

        # self.log("nacrps", nacrps, prog_bar=True)
        self.log("val_masked_nacrps", masked_nacrps, prog_bar=True)

        # self.log("nrmse", nrmse, prog_bar=True)
        self.log("val_masked_nrmse", masked_nrmse, prog_bar=True)

        # Reset metrics
        self.val_masked_crps = 0
        self.val_masked_gt_abs_sum = 0

        self.val_masked_squared_error = 0
        self.val_masked_gt_num = 0
        self.val_masked_gt_min, self.val_masked_gt_max = 1e9, -1e9

    def test_step(self, batch, batch_idx):
        batch_masked_crps, batch_masked_gt_abs_sum, batch_masked_se, batch_masked_gt_min, batch_masked_gt_max, batch_masked_gt_num = self.process_validation_test_batch(
            batch, batch_idx, mode='test'
        )
        self.test_masked_crps += batch_masked_crps
        self.test_masked_gt_abs_sum += batch_masked_gt_abs_sum

        self.test_masked_squared_error += batch_masked_se
        self.test_masked_gt_min = min(self.test_masked_gt_min, batch_masked_gt_min)
        self.test_masked_gt_max = max(self.test_masked_gt_max, batch_masked_gt_max)
        self.test_masked_gt_num += batch_masked_gt_num

    def on_test_epoch_end(self):
        masked_nacrps = self.test_masked_crps / self.test_masked_gt_abs_sum
        masked_nrmse = torch.sqrt(self.test_masked_squared_error / self.test_masked_gt_num) / (self.test_masked_gt_max - self.test_masked_gt_min)

        self.log("test_masked_nacrps", masked_nacrps, prog_bar=True)
        self.log("test_masked_nrmse", masked_nrmse, prog_bar=True)

        # Reset Metrics
        self.test_masked_crps = 0
        self.test_masked_gt_abs_sum = 0

        self.test_masked_squared_error = 0
        self.test_masked_gt_num = 0
        self.test_masked_gt_min, self.test_masked_gt_max = 1e9, -1e9

    def process_validation_test_batch(self, batch, batch_idx, mode):
        context, covariate, horizon, horizon_mask = batch

        batch_size = context.shape[0]
        num_scenarios = self.config.num_scenarios_validation if mode == 'validation' else self.config.num_scenarios_test

        # Context size provided: batch_size x context_length x num_channels=2 x height x width
        # Add a scenario dimension. New size: batch_size x num_scenarios x context_length x num_channels=2 x height x width
        # Where each batch element will have the same context repeated
        # Then merge the batch_size and num_scenarios dimension so that it can be used by ddpm.backward
        context = repeat(context, 'b d c h w -> b s d c h w', s=num_scenarios)
        context = rearrange(context, 'b s d c h w -> (b s) d c h w')

        # Covariate dimension provided: batch_size x context_length x num_features
        # Add a scenario dimension. New size: batch_size x num_scenarios x context_length x num_features
        # Then merge the batch_size and num_scenarios dimension so that it can be used by ddpm.backward
        covariate = repeat(covariate, 'b d f -> b s d f', s=num_scenarios)
        covariate = rearrange(covariate, 'b s d f -> (b s) d f')

        # x = self.diffusion_model.generate_scenarios(context, covariate, self.device)
        x = self.diffusion_model.generate_multistep_scenarios(context, covariate, self.device)

        # print("x.shape:", x.shape)
        # torch.save(x, self.config.store_path + 'predictions.pt')
        # torch.save(horizon, self.config.store_path + 'ground_truth.pt')

        # x dimension: batch_size * num_scenarios x d x c x h x w
        # Resize it to: batch_size x num_scenarios x d x c x h x w
        x = rearrange(x, '(b s) d c h w -> b s d c h w', b=batch_size, s=num_scenarios)
        denormalized_horizon = horizon * self.config.train_std_inundation + self.config.train_mean_inundation
        denormalized_x = x * self.config.train_std_inundation + self.config.train_mean_inundation

        # torch.save(denormalized_x, f"{self.config.store_path}/{mode}_batch_{batch_idx}_predictions.pt")
        # torch.save(denormalized_horizon, f"{self.config.store_path}/{mode}_batch_{batch_idx}_ground_truth.pt")

        # For masked NACRPS and NRMSE
        batch_masked_crps, batch_masked_gt_abs_sum, \
        batch_masked_se, batch_masked_gt_min, \
        batch_masked_gt_max, batch_masked_gt_num = self.get_batch_masked_metric(
            denormalized_x, denormalized_horizon, horizon_mask.bool()
        )

        return batch_masked_crps, batch_masked_gt_abs_sum, batch_masked_se, \
            batch_masked_gt_min, batch_masked_gt_max, batch_masked_gt_num
    
    def get_batch_masked_metric(self, prediction, ground_truth, mask):
        # For CRPS
        x = rearrange(prediction, 'b s d c h w -> (b d c h w) s')
        gt = rearrange(ground_truth, 'b d c h w -> (b d c h w)')
        mask = rearrange(mask, 'b d c h w -> (b d c h w)')

        masked_x = x[mask, :]
        masked_gt = gt[mask]

        masked_crps = ps.crps_ensemble(masked_gt.cpu(), masked_x.cpu())

        # For RMSE
        mean_masked_x = reduce(masked_x, 'v s -> v', 'mean')
        error = torch.square(masked_gt - mean_masked_x).sum()

        return masked_crps.sum(), masked_gt.abs().sum(), error, masked_gt.min(), masked_gt.max(), masked_gt.numel()
