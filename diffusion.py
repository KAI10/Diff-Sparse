import torch
import torch.nn as nn

from einops import rearrange, repeat

from diffusers import UNet2DConditionModel
from hidden_state_net import HiddenStateNet
from patch_embedding import PatchEmbeddingNet
from training_config import TrainingConfig


# DDPM class
class SpatioTemporalDDPM(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        hidden_state_net: HiddenStateNet,
        patch_embedding_net: PatchEmbeddingNet,
        config: TrainingConfig
    ):
        super().__init__()
        self.unet = unet
        self.hidden_state_net = hidden_state_net
        self.patch_embedding_net = patch_embedding_net

        self.config = config
        self.num_diffusion_steps = config.num_diffusion_steps
        self.min_beta = config.min_beta
        self.max_beta = config.max_beta
        self.initialize_diffusion_schedule()
        
    def initialize_diffusion_schedule(self):
        # Number of steps is typically in the order of thousands
        self.betas = torch.linspace(self.min_beta, self.max_beta, self.num_diffusion_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))])

    def forward(self, x0, t, eta):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def get_condition_embedding(self, context, covariate):
        return self.hidden_state_net(context, covariate)

    def get_class_embedding(self, elevation):
        return self.patch_embedding_net(elevation)

    # @torch.compile
    def backward(self, x, condition_embedding, class_embedding, t):
        # x: batch of noisy images (batch_size x num_channels=1 x height x width)
        # context: batch of context for the batch of images (batch_size x context_length x num_channels=2 x height x width)
        # covariate: batch of covariate for the batch of images (batch_size x context_length x num_covariates)
        # t: batch of diffusion timesteps (batch_size x 1)

        if class_embedding is not None:
            predicted_image = self.unet(
                x, timestep=t,
                encoder_hidden_states=condition_embedding, class_labels=class_embedding
            ).sample

        else:
            predicted_image = self.unet(
                x, timestep=t,
                encoder_hidden_states=condition_embedding
            ).sample

        # return predicted_noise
        return predicted_image

    
    def generate_scenarios(self, context, covariate, device):   # Update here for x parameterization
        batch_size = context.shape[0]
        c, h, w = 1, context.shape[-2], context.shape[-1]

        condition_embedding = self.get_condition_embedding(context, covariate)
        if self.config.use_patch_embedding:
            elevation = context[:, 0, 1, :, :]
            class_embedding = self.get_class_embedding(elevation) 
        else:
            class_embedding = None

        x = torch.randn(batch_size, c, h, w, device=device)
        
        for idx, t in enumerate(range(self.num_diffusion_steps-1, -1, -1)):
            time_tensor = (torch.ones(batch_size, device=device) * t).long()

            # x parameterization
            x0_theta = self.backward(x, condition_embedding, class_embedding, time_tensor)

            # print('eta_theta.shape:', eta_theta.shape)
            alpha_t = self.alphas[t]
            alpha_t_bar = self.alpha_bars[t]
            alpha_t_1_bar = self.alpha_bars[t-1] if t > 0 else torch.ones_like(alpha_t_bar, device=device)

            # eta parameterization
            # x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            # x parameterization
            x = (alpha_t.sqrt() * (1 - alpha_t_1_bar) * x + alpha_t_1_bar.sqrt() * (1 - alpha_t) * x0_theta) / (1 - alpha_t_bar)

            if t > 0:
                z = torch.randn(batch_size, c, h, w, device=device)

                # Option 1: sigma_t squared = beta_t
                beta_t = self.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z
                
        return x

    def generate_multistep_scenarios(self, context, covariate, device):
        # context: batch of context for the batch of images (batch_size x context_length x num_channels=2 x height x width)
        # covariate: batch of covariate for the batch of images (batch_size x (context_length+horizon_length) x num_covariates)

        batch_size = context.shape[0]
        c, h, w = 1, context.shape[-2], context.shape[-1]

        context_length = self.config.context_length
        horizon_length = self.config.validation_horizon_length

        # Grabbing additional channels from context.
        # shape: batch_size x num_channels=1 x height x width
        additional_channels = context[:, 0, 1:, :, :]

        context = rearrange(context, 'b d c h w -> d b c h w')
        multistep_prediction = torch.zeros(horizon_length, batch_size, c, h, w, device=device)

        for ht in range(horizon_length):
            # print(f"Predicting horizon timestep {ht+1} ...")
            step_prediction = self.generate_scenarios(
                rearrange(context, 'd b c h w -> b d c h w'),
                covariate[:, ht:ht+context_length, :],
                device
            )

            # save prediction
            # step_prediction shape: batch_size x c x h x w
            multistep_prediction[ht, :, :, :, :] = step_prediction

            # torch.save(context, self.config.store_path + 'context.pt')
            # torch.save(step_prediction, self.config.store_path + 'step_prediction.pt')

            # Append additional channels before appending to context
            step_prediction = torch.cat((step_prediction, additional_channels), dim=1)

            # Append prediction to context for next step
            context = torch.cat(
                # (context, rearrange(step_prediction, 'b c h w -> 1 b c h w')),
                (context[1:, :, :, :, :], rearrange(step_prediction, 'b c h w -> 1 b c h w')),
                dim=0
            )

        multistep_prediction = rearrange(multistep_prediction, 'd b c h w -> b d c h w')
        return multistep_prediction
