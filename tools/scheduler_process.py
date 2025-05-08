import torch
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from typing import List, Optional, Union
# CustomFlowMatchEulerDiscreteScheduler was taken from ostris ai-toolkit trainer:
# https://github.com/ostris/ai-toolkit/blob/9ee1ef2a0a2a9a02b92d114a95f21312e5906e54/toolkit/samplers/custom_flowmatch_sampler.py#L95
class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with torch.no_grad():
            # create weights for timesteps
            num_timesteps = 1000

            # generate the multiplier based on cosmap loss weighing
            # this is only used on linear timesteps for now

            # cosine map weighing is higher in the middle and lower at the ends
            # bot = 1 - 2 * self.sigmas + 2 * self.sigmas ** 2
            # cosmap_weighing = 2 / (math.pi * bot)

            # sigma sqrt weighing is significantly higher at the end and lower at the beginning
            sigma_sqrt_weighing = (self.sigmas**-2.0).float()
            # clip at 1e4 (1e6 is too high)
            sigma_sqrt_weighing = torch.clamp(sigma_sqrt_weighing, max=1e4)
            # bring to a mean of 1
            sigma_sqrt_weighing = sigma_sqrt_weighing / sigma_sqrt_weighing.mean()

            # Create linear timesteps from 1000 to 0
            timesteps = torch.linspace(1000, 0, num_timesteps, device="cpu")

            self.linear_timesteps = timesteps
            # self.linear_timesteps_weights = cosmap_weighing
            self.linear_timesteps_weights = sigma_sqrt_weighing
            
            self.use_dynamic_shifting = False

            # self.sigmas = self.get_sigmas(timesteps, n_dim=1, dtype=torch.float32, device='cpu')
            pass

    def get_weights_for_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Get the indices of the timesteps
        step_indices = [(self.timesteps == t).nonzero().item() for t in timesteps]

        # Get the weights for the timesteps
        weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        ## ref https://github.com/huggingface/diffusers/blob/fbe29c62984c33c6cf9cf7ad120a992fe6d20854/examples/dreambooth/train_dreambooth_sd3.py#L1578
        ## Add noise according to flow matching.
        ## zt = (1 - texp) * x + texp * z1

        # sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        # noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # timestep needs to be in [0, 1], we store them in [0, 1000]
        # noisy_sample = (1 - timestep) * latent + timestep * noise
        t_01 = (timesteps / 1000).to(original_samples.device)
        noisy_model_input = (1 - t_01) * original_samples + t_01 * noise

        # n_dim = original_samples.ndim
        # sigmas = self.get_sigmas(timesteps, n_dim, original_samples.dtype, original_samples.device)
        # noisy_model_input = (1.0 - sigmas) * original_samples + sigmas * noise
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def set_train_timesteps(self, num_timesteps, device, linear=False):
        if linear:
            timesteps = torch.linspace(1000, 0, num_timesteps, device=device)
            self.timesteps = timesteps
            return timesteps
        else:
            # distribute them closer to center. Inference distributes them as a bias toward first
            # Generate values from 0 to 1
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))

            # Scale and reverse the values to go from 1000 to 0
            timesteps = (1 - t) * 1000

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            self.timesteps = timesteps.to(device=device)

            return timesteps