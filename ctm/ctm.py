import math 
from functools import partial
import copy 

import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import einops

from .utils import *
from .models.ctm_mlp import ConsistencyTrajectoryNetwork, Discriminator
from .models.ctm_unet import UNetModel


def ema_eval_wrapper(func):
    def wrapper(self, *args, **kwargs):
        # Swap model parameters with EMA parameters
        model = self.model

        self.model = self.target_model
        
        # Call the original function
        result = func(self, *args, **kwargs)
        
        # Swap the parameters back to the original model
        self.model = model
        return result
    return wrapper


def ema_diffusion_eval_wrapper(func):
    def wrapper(self, *args, **kwargs):
        # Swap model parameters with EMA parameters
        model = self.model
        self.model = self.teacher_model
        # Call the original function
        result = func(self, *args, **kwargs)
        
        # Swap the parameters back to the original model
        self.model = model
        return result
    return wrapper

# https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/script_util.py#L26C1-L53C1
def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        sigma_min=0.002,
        sigma_max=80.0,
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=False,
        weight_schedule="karras",
    )
    return res



class ConsistencyTrajectoryModel(nn.Module):

    def __init__(
            self, 
            data_dim: int,
            cond_dim: int,
            sampler_type: str,
            sigma_data: float,
            sigma_min: float,
            sigma_max: float,
            conditioned: bool,
            device: str,
            use_teacher: bool = False,
            use_gan: bool = False,
            solver_type: str = 'heun',
            n_discrete_t: int = 20,
            lr: float = 1e-4,
            rho: int = 7,
            diffusion_lambda: float = 1.0,
            gan_lambda: float = 0.0,
            ema_rate: float = 0.999,
            n_sampling_steps: int = 10,
            sigma_sample_density_type: str = 'loglogistic',
            datatype: str = '1d', # 1d or image
            num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.use_gan = use_gan
        self.ema_rate = ema_rate
        self.diffusion_lambda = diffusion_lambda
        self.gan_lambda = gan_lambda
        self.n_discrete_t = n_discrete_t
        if datatype == '1d':
            self.model = ConsistencyTrajectoryNetwork(
                x_dim=data_dim,
                hidden_dim=128,
                time_embed_dim=4,
                cond_dim=cond_dim,
                cond_mask_prob=0.0,
                num_hidden_layers=4,
                output_dim=data_dim,
                dropout_rate=0.1,
                cond_conditional=conditioned
            ).to(device)
        else:  # for image type follow CTM: https://github.com/sony/ctm/blob/36c0f57d6cc0cff328f54852e0487e9e4e78f7ce/code/cm/script_util.py#L311
            image_size = data_dim
            defaults = model_and_diffusion_defaults()
            if defaults["channel_mult"] == "":
                if image_size == 512:
                    channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
                elif image_size == 256:
                    channel_mult = (1, 1, 2, 2, 4, 4)
                elif image_size == 128:
                    channel_mult = (1, 1, 2, 3, 4)
                elif image_size == 64:
                    channel_mult = (1, 2, 3, 4)
                elif image_size == 32: # added by zihan
                    channel_mult = (1, 2, 4)
                else:
                    raise ValueError(f"unsupported image size: {image_size}")
            else:
                channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

            attention_ds = []
            for res in defaults["attention_resolutions"].split(","):
                attention_ds.append(image_size // int(res))

            self.model=UNetModel(
                image_size,
                in_channels=3,
                model_channels=defaults["num_channels"],
                out_channels=(3 if not defaults["learn_sigma"] else 6),
                num_res_blocks=defaults["num_res_blocks"],
                attention_resolutions=tuple(attention_ds),
                dropout=defaults["dropout"],
                channel_mult=channel_mult,
                # num_classes=(NUM_CLASSES if defaults["class_cond"] else None),
                num_classes=(num_classes if conditioned else None),
                use_checkpoint=defaults["use_checkpoint"],
                use_fp16=defaults["use_fp16"],
                num_heads=defaults["num_heads"],
                num_heads_upsample=defaults["num_heads_upsample"],
                num_head_channels=defaults["num_head_channels"],
                use_scale_shift_norm=defaults["use_scale_shift_norm"],
                resblock_updown=defaults["resblock_updown"],
                use_new_attention_order=defaults["use_new_attention_order"],
            ).to(device)

        # we need an ema version of the model for the consistency loss
        self.target_model = copy.deepcopy(self.model)
        for param in self.target_model.parameters():
            param.requires_grad = False
        # we further can use a teacher model for the solver
        self.use_teacher = use_teacher
        if self.use_teacher:
            self.teacher_model = copy.deepcopy(self.model)
        self.device = device
        self.sampler_type = sampler_type
        # use the score wrapper 
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.n_sampling_steps = n_sampling_steps
        self.solver_type = solver_type
        self.sigma_sample_density_type = sigma_sample_density_type
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = 0
        self.discriminator = Discriminator(input_dim=data_dim, hidden_dim=256, num_hidden_layers=4, output_dim=1, dropout_rate=0.0).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.002)
        self.criterion = nn.BCELoss()
        
    def diffusion_wrapper(self, model, x, cond, t, s):
        """
        Performs the diffusion wrapper for the given model, x, cond, and t.
        Based on the conditioning from EDM Karras et al. 2022.

        Args:
            model (torch.nn.Module): The neural network model to be used for the diffusion process.
            x (torch.Tensor): The input tensor to the model.
            cond (torch.Tensor): The conditioning tensor to be used during the diffusion process.
            t (float): The time step for the diffusion process.

        Returns:
            torch.Tensor: The scaled output tensor after applying the diffusion wrapper to the model.
        """
        c_skip = self.sigma_data**2 / (
            t ** 2 + self.sigma_data**2
        )
        c_out = (
            t * self.sigma_data / (t**2 + self.sigma_data**2) ** 0.5
        )
        # these two are not mentioned in the paper but they use it in their code
        c_in = 1 / (t**2 + self.sigma_data**2) ** 0.5
        
        t = 0.25 * torch.log(t + 1e-40)
        c_in = append_dims(c_in, x.ndim)
        c_out = append_dims(c_out, x.ndim)
        c_skip = append_dims(c_skip, x.ndim)

        if len(x.shape) > 2 and len(t.shape) < 1:  # for image
            # expand t to batch size
            t = t.repeat(x.shape[0])
            s = s.repeat(x.shape[0])

        diffusion_output = model(c_in * x, cond, t, s)
        scaled_output = c_out * diffusion_output + c_skip * x
        
        return scaled_output
    
    def ctm_wrapper(self, model, x, cond, t, s):
        """
        Applies the new ctm wrapper from page 4 of https://openreview.net/attachment?id=ymjI8feDTD&name=pdf

        Args:
            model (torch.nn.Module): The neural network model to be used for the diffusion process.
            x (torch.Tensor): The input tensor to the model.
            cond (torch.Tensor): The conditioning tensor to be used during the diffusion process.
            t (float): The time step for the diffusion process.
            s: (float): the target noise level for the diffusion process.

        Returns:
            torch.Tensor: The scaled output tensor after applying the diffusion wrapper to the model.
        """
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        if len(s.shape) == 1:
            s = s.unsqueeze(1)

        c1 = (s / t)
        c2 = (1 - s /t)
        c1 = append_dims(c1, x.ndim)
        c2 = append_dims(c2, x.ndim)

        G_0 =  c1 * x +  c2 * self.diffusion_wrapper(model, x, cond, t, s)
        
        return G_0
    
    def _update_ema_weights(self):
        """
        Updates the exponential moving average (EMA) weights of the target model.

        The method performs the following steps:
        1. Gets the state dictionary of the self.model (source model).
        2. Updates the EMA weights for each parameter in the target model by computing the weighted average between 
        the corresponding parameter in the target model and the parameter in the source model, using the EMA rate parameter.
        """
        # Get the state dictionary of the current/source model
        state_dict = self.model.state_dict()
        # Get the state dictionary of the target model
        target_state_dict = self.target_model.state_dict()

        # Iterate over the parameters in the target model state dictionary
        for key in state_dict:
            if key in target_state_dict:
                # Update the EMA weights for each parameter
                target_param_data = target_state_dict[key].data
                model_param_data = state_dict[key].data
                # target_state_dict[key].data.copy_((1 - self.ema_rate) * target_param_data + self.ema_rate * model_param_data)
                target_state_dict[key].data.copy_(self.ema_rate * target_param_data + (1 - self.ema_rate) * model_param_data)

        # You can optionally load the updated state dict into the target model, if necessary
        # self.target_model.load_state_dict(target_state_dict)

    # use original
    # def train_step(self, x, cond, train_step, max_steps):
    #     """
    #     Main training step method to compute the loss for the Consistency Trajectory Model.
    #     The loss consists of three parts: the consistency loss, the diffusion loss, and the GAN loss (optional).
    #     The first part is similar to Song et al. 23 and the second part is similar to Karras et al. 2022.

    #     Args:

    #     Returns:

    #     """
    #     self.model.train()
    #     t = self.make_sample_density()(shape=(len(x),), device=self.device)
    #     noise = torch.randn_like(x)
    #     # next we sample s in range of [0, t]
    #     s = torch.rand_like(t) * t
    #     # next we sample u in range of (s, t]
    #     u = torch.rand_like(t) * (t - s) + s
    #     # get the noise samples
    #     x_t = x + noise * append_dims(t, x.ndim)
    #     # use the solver if we have a teacher model otherwise use the euler method
    #     solver_target = self.solver(x_t, cond, t, u)

    #     # compute the ctm consistency loss
    #     ctm_loss = self.ctm_loss(x_t, cond, t, s, u, solver_target)
        
    #     # compute the diffusion loss
    #     diffusion_loss = self.diffusion_loss(x, x_t, cond, t)

    #     # compute the GAN loss if chosen
    #     if self.use_gan:
    #         gan_loss = self.gan_loss(x, x_t, cond, t)
    #         if train_step < 0.3*max_steps: # warm-up, train discriminator only
    #             gan_lambda = 0
    #         else:
    #             gan_lambda = self.gan_lambda
    #     else:
    #             gan_lambda = 0
    #             gan_loss = 0
    
    #     # compute the total loss
    #     loss = ctm_loss + self.diffusion_lambda * diffusion_loss + gan_lambda * gan_loss

    #     # perform the backward pass
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     # update the ema weights
    #     self._update_ema_weights()
        
    #     return loss, ctm_loss, diffusion_loss, gan_loss

    def train_step(self, x, cond, train_step, max_steps):
        """
        Main training step method to compute the loss for the Consistency Trajectory Model.
        The loss consists of three parts: the consistency loss, the diffusion loss, and the GAN loss (optional).
        The first part is similar to Song et al. (2023) and the second part is similar to Karras et al. (2022).
        The GAN Part is not implemented right now, since its not attractive for Imitation Learning applications.
        """
        self.model.train()
        t_ctm, s, u = self.sample_noise_levels(shape=(len(x),), N=self.n_discrete_t, device=self.device)
        noise = torch.randn_like(x).to(self.device)
        # get the noise samples
        x_t = x + noise * append_dims(t_ctm, x.ndim)
        # use the solver if we have a teacher model otherwise use the euler method
        solver_target = self.solver(x_t, cond, t_ctm, u).detach()

        # compute the ctm consistency loss
        ctm_loss = self.ctm_loss(x_t, cond, t_ctm, s, u, solver_target)
        
        # compute the diffusion loss
        # sample noise for the diffusion loss from the continuous noise distribution
        if self.diffusion_lambda > 0:
            t_sm = self.make_sample_density()(shape=(len(x),), device=self.device)
            x_t_sm = x + noise * append_dims(t_sm, x.ndim)
            diffusion_loss = self.diffusion_loss(x, x_t_sm, cond, t_sm)
        else:
            diffusion_loss = 0

        # compute the GAN loss if chosen
        # not implemented yet
        if self.use_gan:
            gan_loss = self.gan_loss(x, x_t, cond, t_ctm)
            if train_step < 0.3*max_steps: # warm-up, train discriminator only
                gan_lambda = 0
            else:
                gan_lambda = self.gan_lambda
        else:
            gan_lambda = 0
            gan_loss = 0

        # compute the total loss
        loss = ctm_loss + self.diffusion_lambda * diffusion_loss + gan_lambda * gan_loss
        
        # perform the backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update the ema weights
        self._update_ema_weights()

        # self.target_model.load_state_dict(self.model.state_dict())  # improved techniques of CM for using no target (same as model)
        
        return loss, ctm_loss, diffusion_loss, gan_loss

    def gan_loss(self, x, x_t, cond, t):
        jump_target = einops.repeat(torch.tensor([0]), '1 -> (b 1)', b=len(x_t)).to(x_t.device)
        ctm_pred = self.ctm_wrapper(self.model, x_t, cond, t, jump_target)

        batch_size = x.shape[0]
        real_labels = torch.ones(batch_size, 1).to(x.device)
        fake_labels = torch.zeros(batch_size, 1).to(x.device)

        # discriminator loss
        self.discriminator_optimizer.zero_grad()
        real_loss = self.criterion(self.discriminator(x), real_labels)
        fake_loss = self.criterion(self.discriminator(ctm_pred.detach()), fake_labels)
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        # generator loss
        generator_loss = self.criterion(self.discriminator(ctm_pred), real_labels)
        return generator_loss
    
    def sample_noise_levels(self, shape, N, device='cpu'):
        """
        Samples a tensor of the specified shape with noise levels 
        from `N` discretized levels of the noise scheduler.

        Args:
            shape (tuple): Shape of the tensor to sample.
            N (int): Number of discrete noise levels to discretize the scheduler.
            device (str): Device on which to create the noise levels, 'cpu' or 'cuda'.

        Returns:
            torch.Tensor: Tensor containing sampled noise levels.
        """
        # Get the N discretized noise levels
        discretized_sigmas = get_sigmas_exponential(N, self.sigma_min, self.sigma_max, self.device)
        
        # Sample indices from this discretized range
        t = torch.randint(1, N, size=shape, device=device)
        s = torch.round(torch.rand_like(t.to(torch.float32)) * t.to(torch.float32)).to(torch.int64).to(device)
        u = torch.round(torch.rand_like(t.to(torch.float32)) * (t.to(torch.float32) -1  - s.to(torch.float32))+ s).to(torch.int64).to(device)

        # Use these indices to gather the noise levels from the discretized sigmas
        sigma_t = discretized_sigmas[t]
        sigma_s = discretized_sigmas[s]
        sigma_u = discretized_sigmas[u]
        return sigma_t, sigma_s, sigma_u

    def solver(self, x, cond, t, s):
        """
        Eq. (3) in the paper
        """
        if self.use_teacher:
            solver = self.teacher_model
        else:
            solver = self.model

        if self.solver_type == 'euler':
            solver_pred = self.euler_update_step(solver, x, cond, t, s)
        elif self.solver_type == 'heun':
            solver_pred = self.heun_update_step(solver, x, cond, t, s)
        elif self.solver_type == 'ddim':
            solver_pred = self.ddim_update_step(solver, x, cond, t, s)

        return solver_pred

    
    def eval_step(self, x, cond):
        """
        Eval step method to compute the loss for the action prediction.
        """
        self.model.eval()
        self.target_model.eval()
        x = x.to(self.device)
        cond = cond.to(self.device)
        # next generate the discrete timesteps
        t = [self.sample_discrete_timesteps(i) for i in range(self.t_steps)]
        # compute the loss
        x_T = torch.randn_like(x) * self.sigma_max
        pred_x = self. sample(x_T, cond, t)
        loss = torch.nn.functional.mse_loss(pred_x, x)
        return loss
    
    def ctm_loss(self, x_t, cond, t, s, u, solver_target):
        """
        # TODO add description

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim].
            cond (torch.Tensor): Conditioning tensor of shape [batch_size, cond_dim].
            t1 (torch.Tensor): First discrete timestep tensor of shape [batch_size, 1].
            t2 (torch.Tensor): Second discrete timestep tensor of shape [batch_size, 1].

        Returns:
            torch.Tensor: Consistency loss tensor of shape [].
        """
        ## TODO: zihan commented
        jump_target = einops.repeat(torch.tensor([0]), '1 -> (b 1)', b=len(x_t)).to(s.device)
        # compute the ctm prediction: jump from t to s
        ctm_pred = self.ctm_wrapper(self.model, x_t, cond, t, s)

        # compute the ctm target prediction with ema parameters inside self.target_model: jump from u to s
        with torch.no_grad():
            ctm_target = self.ctm_wrapper(self.target_model, solver_target, cond, u, s)
            ctm_target_clean = self.ctm_wrapper(self.target_model, ctm_target, cond, s, jump_target)

        # transform them into the clean data space by jumping without gradient from s to 0
        # for both predictions and comparing them in the clean data space
        """
        # Compute f(f(x)) with gradient tracking for the inside f only
        inner_result = f(x)  # First application of f, gradients are tracked here
        inner_result_detached = inner_result.detach().requires_grad_()  # Detach and enable grad for outer f
        outer_result = f(inner_result_detached)  # Second application of f, gradients not tracked
        to check if this is right (zihan)
        """
        # with torch.no_grad():
        ctm_pred_clean = self.ctm_wrapper(self.target_model, ctm_pred, cond, s, jump_target)  # this one is better
        # ctm_pred_clean = self.ctm_wrapper(copy.deepcopy(self.model), ctm_pred, cond, s, jump_target)
        
        # compute the ctm loss
        ctm_loss = torch.nn.functional.mse_loss(ctm_pred_clean, ctm_target_clean)

        # # using previous code
        # ctm_pred = self.ctm_wrapper(self.model, x_t, cond, t, s)

        # # compute the ctm target prediction without gradient
        # with torch.no_grad():
        #     ctm_target = self.ctm_wrapper(self.target_model, solver_target, cond, u, s)

        # # compute the ctm loss
        # ctm_loss = torch.nn.functional.mse_loss(ctm_pred, ctm_target)

        return ctm_loss


    @torch.no_grad()   
    def heun_update_step(self, model, x, cond, t1, t2):
        """
        Computes a single Heun update step from the Euler sampler with the teacher model

        Parameters:
        x (torch.Tensor): The input tensor.
        t1 (torch.Tensor): The initial timestep.
        t2 (torch.Tensor): The final timestep.
        x0 (torch.Tensor): The ground truth value used to compute the Euler update step.

        Returns:
        torch.Tensor: The output tensor after taking the Euler update step.
        """
        denoised = self.ctm_wrapper(model, x, cond, t1, t1)
        d = (x - denoised) / append_dims(t1, x.ndim)
        
        
        sample_temp = x + d * append_dims(t2 - t1, x.ndim)
        denoised_2 = self.ctm_wrapper(model, sample_temp, cond, t2, t2)
        d_2 = (sample_temp - denoised_2) / append_dims(t2, x.ndim)
        d_prime = (d + d_2) / 2
        samples = x + d_prime * append_dims(t2 - t1, x.ndim)
        
        return samples
    
    @torch.no_grad()   
    def ddim_update_step(self, model, x, cond, t1, t2):
        """
        Computes a single Heun update step from the DDIM sampler with the teacher model

        Parameters:
        x (torch.Tensor): The input tensor.
        t1 (torch.Tensor): The initial timestep.
        t2 (torch.Tensor): The final timestep.
        x0 (torch.Tensor): The ground truth value used to compute the Euler update step.

        Returns:
        torch.Tensor: The output tensor after taking the Euler update step.
        """
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        denoised = self.ctm_wrapper(model, x, cond, t1, t1)
        
        t, t_next = t_fn(t1), t_fn(t2)
        h = append_dims(t_next - t, x.ndim)
        samples = append_dims((sigma_fn(t_next) / sigma_fn(t)), x.ndim) * x - (-h).expm1() * denoised
        
        return samples

    def get_diffusion_scalings(self, sigma):
        """
        Computes the scaling factors for diffusion training at a given time step sigma.

        Args:
        - self: the object instance of the model
        - sigma (float or torch.Tensor): the time step at which to compute the scaling factors
        
        , where self.sigma_data: the data noise level of the diffusion process, set during initialization of the model

        Returns:
        - c_skip (torch.Tensor): the scaling factor for skipping the diffusion model for the given time step sigma
        - c_out (torch.Tensor): the scaling factor for the output of the diffusion model for the given time step sigma
        - c_in (torch.Tensor): the scaling factor for the input of the diffusion model for the given time step sigma

        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in
    
    @staticmethod
    def mean_flat(tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def diffusion_train_step(self,  x, cond, train_step, max_steps):
        """
        Computes the training loss and performs a single update step for the score-based model.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, dim)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)

        Returns:
        - loss.item() (float): the scalar value of the training loss for this batch

        """
        self.model.train()
        x = x.to(self.device)
        cond = cond.to(self.device)
        self.optimizer.zero_grad()
        noise = torch.randn_like(x).to(self.device)
        t_sm = self.make_sample_density()(shape=(len(x),), device=self.device)
        x_t_sm = x + noise * append_dims(t_sm, x.ndim)
        loss = self.diffusion_loss(x, x_t_sm, cond, t_sm)
        loss.backward()
        self.optimizer.step()

        self._update_ema_weights()

        return loss.item()

    
    def diffusion_loss(self, x, x_t, cond, t):
        """
        Computes the diffusion training loss for the given model, input, condition, and time.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, channels, height, width)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)
        - t (torch.Tensor): the time step tensor of shape (batch_size,)

        Returns:
        - loss (torch.Tensor): the diffusion training loss tensor of shape ()

        The diffusion training loss is computed based on the following equation from Karras et al. 2022:
        loss = (model_output - target)^2.mean()
        where,
        - noise: a tensor of the same shape as x, containing randomly sampled noise
        - x_t: a tensor of the same shape as x, obtained by adding the noise tensor to x
        - c_skip, c_out, c_in: scaling tensors obtained from the diffusion scalings for the given time step
        - t: a tensor of the same shape as t, obtained by taking the natural logarithm of t and dividing it by 4
        - model_output: the output tensor of the model for the input x_1, condition cond, and time t
        - target: the target tensor for the given input x, scaling tensors c_skip, c_out, c_in, and time t
        """
        c_skip, c_out, c_in = [append_dims(c, x.ndim) for c in self.get_diffusion_scalings(t)]
        t = torch.log(t) / 4
        model_output = self.model(x_t * c_in, cond, t, t)
        target = (x - c_skip * x_t) / c_out
        return (model_output - target).pow(2).mean()
        
    def update_teacher_model(self):     
        # self.teacher_model.load_state_dict(self.target_model.state_dict()) # target model is not updated in diffusion training
        self.teacher_model.load_state_dict(self.model.state_dict())
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # next we init the model and target model with the same weights from the teacher
        self.model.load_state_dict(self.teacher_model.state_dict())
        for param in self.model.parameters():
            param.requires_grad = True
        self.target_model.load_state_dict(self.teacher_model.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False
        print('Updated Teacher Model and froze all parameters!')
        
    def euler_update_step(self, x, t1, t2, denoised):
        """
        Computes a single update step from the Euler sampler with a ground truth value.

        Parameters:
        x (torch.Tensor): The input tensor.
        t1 (torch.Tensor): The initial timestep.
        t2 (torch.Tensor): The final timestep.
        x0 (torch.Tensor): The ground truth value used to compute the Euler update step.

        Returns:
        torch.Tensor: The output tensor after taking the Euler update step.
        """
        d = (x - denoised) / append_dims(t1, x.ndim)
        samples = x + d * append_dims(t2 - t1, x.ndim)
        return samples
    
    def euler_single_step(self, model, x, cond, t1, t2):
        """
        
        """
        denoised = self.diffusion_wrapper(model, x, cond, t1, t1)
        d = (x - denoised) / append_dims(t1, x.ndim)
        samples = x + d * append_dims(t2 - t1, x.ndim)
        return samples

    @torch.no_grad()
    @ema_eval_wrapper
    def sample_singlestep(self, x_shape, cond, return_seq=False):
        """
        Samples a single step from the trained consistency trajectory model. 
        If return_seq is True, returns a list of sampled tensors, 
        otherwise returns a single tensor. 
        
        Args:
        - x_shape (tuple): the shape of the tensor to be sampled.
        - cond (torch.Tensor or None): the conditional tensor.
        - return_seq (bool, optional): whether to return a list of sampled tensors (default False).
        
        Returns:
        - (torch.Tensor or list): the sampled tensor(s).
        """
        sampled_x = []
        self.model.eval()
        if cond is not None:
            cond = cond.to(self.device)

        x = torch.randn_like(x_shape).to(self.device) * self.sigma_max
        sampled_x.append(x)
        x = self.ctm_wrapper(self.model, x, cond, torch.tensor([self.sigma_max]).to(x.device), torch.tensor([0]).to(x.device))
        sampled_x.append(x)
        if return_seq:
            return sampled_x
        else:
            return x
        
    @torch.no_grad()
    @ema_eval_wrapper
    def sample_diffusion_euler(self, x_shape, cond, n_sampling_steps=None, return_seq=False):
        """
        Sample from the pre-trained diffusion model using the Euler method. This method is used for sanity checking 
        the learned diffusion model. It generates a sequence of samples by taking small steps from one sample to the next. 
        At each step, it generates a new noise from a normal distribution and combines it with the previous sample 
        to get the next sample.
        
        Parameters:
        - x_shape (torch.Tensor): Shape of the input tensor to the model.
        - cond (torch.Tensor): Conditional information for the model.
        - n_sampling_steps (int, optional): Number of sampling steps to take. Defaults to None.
        - return_seq (bool, optional): Whether to return the full sequence of samples or just the final one. 
                                        Defaults to False.
                                        
        Returns:
        - x (torch.Tensor or List[torch.Tensor]): Sampled tensor from the model. If `return_seq=True`, it returns
                                                a list of tensors, otherwise it returns a single tensor.
        """
        self.model.eval()
        if cond is not None:
            cond = cond.to(self.device)
        x = torch.randn_like(x_shape).to(self.device) * self.sigma_max 
        # x = torch.linspace(-4, 4, len(x_shape)).view(len(x_shape), 1).to(self.device)

        sampled_x = []
        if n_sampling_steps is None:
            n_sampling_steps = self.n_sampling_steps
        
        # sample the sequence of timesteps
        sigmas = self.sample_seq_timesteps(N=n_sampling_steps, type='exponential')
        sampled_x.append(x)
        # iterate over the remaining timesteps
        for i in trange(len(sigmas) - 1, disable=True):
            denoised = self.diffusion_wrapper(self.model, x, cond, sigmas[i], sigmas[i])
            x = self.euler_update_step(x, sigmas[i], sigmas[i+1], denoised)
            sampled_x.append(x)
        if return_seq:
            return sampled_x
        else:
            return x
    
    @torch.no_grad()
    @ema_eval_wrapper
    def ctm_gamma_sampler(self, x_shape, cond, gamma, n_sampling_steps=None, return_seq=False):
        """
        Alg. 3 in the paper of CTM (page 22)
        """
        if isinstance(gamma, float):
            gamma = torch.tensor([gamma])
        gamma = gamma.to(self.device)

        self.model.eval()
        if cond is not None:
            cond = cond.to(self.device)
        x = torch.randn_like(x_shape).to(self.device) * self.sigma_max
        # x = torch.linspace(-4, 4, len(x_shape)).view(len(x_shape), 1).to(self.device)

        sampled_x = []
        if n_sampling_steps is None:
            n_sampling_steps = self.n_sampling_steps
        
        # sample the sequence of timesteps
        sigmas = self.sample_seq_timesteps(N=n_sampling_steps, type='exponential')
        sampled_x.append(x)
        # iterate over the remaining timesteps
        for i in trange(len(sigmas) - 1, disable=True):
            # get the new sigma value 
            sigma_hat = sigmas[i+1] * torch.sqrt(1 - gamma.squeeze() ** 2)
            # get the denoised value
            x_t_gamma = self.ctm_wrapper(self.model, x, cond, sigmas[i], sigma_hat)
            
            if sigmas[i + 1] > 0:
                x = x_t_gamma + gamma * sigmas[i+1] * torch.randn_like(x_shape).to(self.device)
            
            sampled_x.append(x)
        if return_seq:
            return sampled_x
        else:
            return x

    def sample_seq_timesteps(self, N=100, type='karras'):
        """
        Generates a sequence of N timesteps for the given type.

        Args:
        - self: the object instance of the model
        - N (int): the number of timesteps to generate
        - type (str): the type of sequence to generate, either 'karras', 'linear', or 'exponential'

        Returns:
        - t (torch.Tensor): the generated sequence of timesteps of shape (N,)

        The method generates a sequence of timesteps for the given type using one of the following functions:
        - get_sigmas_karras: a function that generates a sequence of timesteps using the Karras et al. schedule
        - get_sigmas_linear: a function that generates a sequence of timesteps linearly spaced between sigma_min and sigma_max
        - get_sigmas_exponential: a function that generates a sequence of timesteps exponentially spaced between sigma_min and sigma_max
        where,
        - self.sigma_min, self.sigma_max: the minimum and maximum timesteps, set during initialization of the model
        - self.rho: the decay rate for the Karras et al. schedule, set during initialization of the model
        - self.device: the device on which to generate the timesteps, set during initialization of the model

        """
        if type == 'karras':
            t = get_sigmas_karras(N, self.sigma_min, self.sigma_max, self.rho, self.device)
        elif type == 'linear':
            t = get_sigmas_linear(N, self.sigma_min, self.sigma_max, self.device)
        elif type == 'exponential':
            t = get_sigmas_exponential(N, self.sigma_min, self.sigma_max, self.device)
        else:
            raise NotImplementedError('Chosen Scheduler is implemented!')
        return t
    
    def make_sample_density(self):
        """
        Returns a function that generates random timesteps based on the chosen sample density.

        Args:
        - self: the object instance of the model

        Returns:
        - sample_density_fn (callable): a function that generates random timesteps

        The method returns a callable function that generates random timesteps based on the chosen sample density.
        The available sample densities are:
        - 'lognormal': generates random timesteps from a log-normal distribution with mean and standard deviation set
                    during initialization of the model also used in Karras et al. (2022)
        - 'loglogistic': generates random timesteps from a log-logistic distribution with location parameter set to the
                        natural logarithm of the sigma_data parameter and scale and range parameters set during initialization
                        of the model
        - 'loguniform': generates random timesteps from a log-uniform distribution with range parameters set during
                        initialization of the model
        - 'uniform': generates random timesteps from a uniform distribution with range parameters set during initialization
                    of the model
        - 'v-diffusion': generates random timesteps using the Variational Diffusion sampler with range parameters set during
                        initialization of the model
        - 'discrete': generates random timesteps from the noise schedule using the exponential density
        - 'split-lognormal': generates random timesteps from a split log-normal distribution with mean and standard deviation
                            set during initialization of the model
        """
        sd_config = []
        
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(rand_log_normal, loc=loc, scale=scale)
        
        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_uniform, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'uniform':
            return partial(rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.n_sampling_steps, 'exponential')
            return partial(rand_discrete, values=sigmas)
        else:
            raise ValueError('Unknown sample density type')
    
    

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = ((max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho) # [:-1]
    return sigmas.to(device)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]