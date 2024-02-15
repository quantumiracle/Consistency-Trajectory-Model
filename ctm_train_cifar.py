
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose, Normalize
from torch.utils.data import DataLoader
from ctm.ctm import ConsistencyTrajectoryModel
from ctm.toy_tasks.data_generator import DataGenerator
from ctm.visualization.vis_utils import plot_main_figure, plot_images
from data import get_dataset


"""
Discrete consistency distillation training of the consistency model on a toy task.
We train a diffusion model and the consistency model at the same time and iteratively 
update the weights of the consistency model and the diffusion model.
"""

if __name__ == "__main__":

    device = 'cuda'  # 'cpu'
    conditioned = False # whether to use conditional training
    n_sampling_steps = 10
    use_pretraining = True
    plot_n_samples = 10

    train_epochs = 2000
    # chose one of the following toy tasks: 'three_gmm_1D' 'uneven_two_gmm_1D' 'two_gmm_1D' 'single_gaussian_1D'
    # data_manager = DataGenerator('two_gmm_1D')
    # samples, cond = data_manager.generate_samples(5000)
    # samples = samples.reshape(-1, 1).to(device)

    evaluation = False
    drop_last = True # If `True`, drop the last batch if it is smaller than the batch size. Default is `True`; if `False`, the last batch will be padded with zeros and a mask will be returned.
    batch_size = 128

    train_dataloader = DataLoader(
        get_dataset('cifar10', train=True, evaluation=evaluation), 
        batch_size=batch_size, 
        shuffle=not evaluation, 
        num_workers=16, 
        pin_memory=True, 
        drop_last=drop_last,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        get_dataset('cifar10', train=False, evaluation=evaluation), 
        batch_size=batch_size, 
        shuffle=not evaluation, 
        num_workers=16, 
        pin_memory=True, 
        drop_last=drop_last,
        persistent_workers=True,
    )

    # get image size from dataset
    image_size = next(iter(train_dataloader))[0].shape[-1]

    ctm = ConsistencyTrajectoryModel(
        data_dim=image_size,
        cond_dim=1,
        sampler_type='euler',
        lr=4e-4,  # 1e-3
        sigma_data=0.5,  # https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/script_util.py#L95
        sigma_min=0.002,
        sigma_max=80.0, # 1; choose according to task data distribution
        solver_type='heun',
        n_discrete_t=18,
        conditioned=conditioned,
        diffusion_lambda= 1,
        use_gan=False,
        gan_lambda= 1,
        device=device,
        rho=7,
        ema_rate=0.999,
        n_sampling_steps=n_sampling_steps,
        use_teacher=use_pretraining,
        datatype='image'
    )

    # if not simultanous_training:
    # First pretrain the diffusion model and then train the consistency model
    if use_pretraining:
        for i in range(train_epochs):
            pbar = tqdm(train_dataloader)
            for samples, cond in pbar:
                samples = samples.to(device)
                cond = cond.reshape(-1, 1).to(device)  
                diff_loss = ctm.diffusion_train_step(samples, cond, i, train_epochs)
                pbar.set_description(f"Step {i}, Diff Loss: {diff_loss:.8f}")
        
        ctm.update_teacher_model()
        
        plot_main_figure(
            ctm, 
            plot_n_samples,
            train_epochs, 
            sampling_method='euler', 
            n_sampling_steps=n_sampling_steps,
            save_path='./plots/'
        )
    

    # Train the consistency trajectory model either simultanously with the diffusion model or after pretraining
    for i in range(train_epochs):
        pbar = tqdm(train_dataloader)
        for samples, cond in pbar:
            samples = samples.to(device)
            cond = cond.reshape(-1, 1).to(device)     
            loss, ctm_loss, diffusion_loss, gan_loss = ctm.train_step(samples, cond, i, train_epochs)
            pbar.set_description(f"Step {i}, Loss: {loss:.4f}, CTM Loss: {ctm_loss:.4f}, Diff Loss: {diffusion_loss:.4f}, GAN Loss: {gan_loss:.4f}")
            # pbar.update(1)
    
    # Plotting the results of the training
    # We do this for the one-step and the multi-step sampler to compare the results
    if not use_pretraining:
            plot_images(
                ctm, 
                plot_n_samples,
                train_epochs, 
                sampling_method='euler', 
                n_sampling_steps=n_sampling_steps,
                save_path='./plots/'
            )
    
    plot_images(
        ctm, 
        plot_n_samples,
        train_epochs, 
        sampling_method='onestep', 
        n_sampling_steps=n_sampling_steps,
        save_path='./plots/ctm'
    )

    plot_images(
        ctm, 
        plot_n_samples,
        train_epochs, 
        sampling_method='multistep', 
        n_sampling_steps=n_sampling_steps,
        save_path='./plots/ctm'
    )

            
    print('done')