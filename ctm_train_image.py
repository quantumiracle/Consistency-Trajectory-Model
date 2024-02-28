import torch
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose, Normalize
from torch.utils.data import DataLoader
from ctm.ctm import ConsistencyTrajectoryModel
from ctm.toy_tasks.data_generator import DataGenerator
from ctm.visualization.vis_utils import plot_main_figure, plot_images, sample_images
from ctm.eval import eval
from data import get_dataset


"""
Discrete consistency distillation training of the consistency model on a toy task.
We train a diffusion model and the consistency model at the same time and iteratively 
update the weights of the consistency model and the diffusion model.
"""

def eval_model(model, dataset, image_shape, num_samples=1000, n_sampling_steps=10, sample_dir='./plots/eval', conditioned=False, num_classes=10):
    sample_images(
    model,
    image_shape,
    num_samples,
    sampling_method='euler', 
    n_sampling_steps=n_sampling_steps,
    save_path=sample_dir,
    conditioned=conditioned,
    num_classes=num_classes,
    )
    eval(sample_dir, data_name=dataset, metric='fid', eval_num_samples=num_samples, delete=True, out=False)
            

if __name__ == "__main__":

    device = 'cuda:3'  # 'cpu'
    torch.cuda.set_device(device)
    print(torch.cuda.current_device())

    dataset = ['cifar10',  'imagenet64'][1]

    hyperparameters = {
        'cifar10': {
            'learning_rate': 4e-4,
            'n_discrete_t': 18,
            'ema_rate': 0.999,
            'solver': 'heun',
            'total_train_iters': 100000,
            'batch_size': 256,
            'image_size': 32,
            'num_classes': 10,
        },
        'imagenet64': {
            'learning_rate': 8e-6,  # for using solver case
            'n_discrete_t': 40,
            'ema_rate': 0.999,
            'solver': 'heun',
            'total_train_iters': int(30000*2048/64),  # use different batchsize from paper
            'batch_size': 64, # larger batch memory is insufficient 
            'image_size': 64,
            'num_classes': 1000,
        }
    }

    conditioned = True # whether to use conditional training
    n_sampling_steps = 10
    use_pretraining = False
    plot_n_samples = 10
    eval_interval = 1000

    evaluation = False
    drop_last = True # If `True`, drop the last batch if it is smaller than the batch size. Default is `True`; if `False`, the last batch will be padded with zeros and a mask will be returned.
    eval_fid = True
    plot_dir = f'./plots/ctm_{dataset}'

    if dataset == 'imagenet64':  
        from ctm.image_datasets import load_data
        # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/image_datasets.py
        train_dataloader = load_data(
                data_dir='tmp/train/',  # ctm is not using downsampled Imagenet64 directly for training, but using ILSVRC2012
                batch_size=hyperparameters[dataset]['batch_size'],  # 128 memory not enough
                image_size=hyperparameters[dataset]['image_size'],
                class_cond=conditioned,
                data_name=dataset,
                use_MPI=False,
                # device_id = '1'
            )
        # val_dataloader = load_data(
        #         data_dir='tmp/imagenet64/val',
        #         batch_size=batch_size,
        #         image_size=64,
        #         class_cond=conditioned,
        #     )
    elif dataset == 'cifar10':
        train_dataloader = DataLoader(
            get_dataset(dataset, train=True, evaluation=evaluation), 
            batch_size=hyperparameters[dataset]['batch_size'], 
            shuffle=not evaluation, 
            num_workers=16, 
            pin_memory=True, 
            drop_last=drop_last,
            persistent_workers=True,
        )

        # val_dataloader = DataLoader(
        #     get_dataset(dataset, train=False, evaluation=evaluation), 
        #     batch_size=batch_size, 
        #     shuffle=not evaluation, 
        #     num_workers=16, 
        #     pin_memory=True, 
        #     drop_last=drop_last,
        #     persistent_workers=True,
        # )
    else:
        raise NotImplementedError

    # get image size from dataset
    example_image = next(iter(train_dataloader))[0]  # batch data: (batch, channel, H, W); cond
    image_size = example_image.shape[-1]
    image_shape = example_image.shape[1:]  
    print('shape: ', image_shape)

    ctm = ConsistencyTrajectoryModel(
        data_dim=image_size,
        cond_dim=1,
        sampler_type='euler',
        lr=hyperparameters[dataset]['learning_rate'],  # 1e-3
        sigma_data=0.5,  # https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/script_util.py#L95
        sigma_min=0.002,
        sigma_max=80.0, # 1; choose according to task data distribution
        solver_type=hyperparameters[dataset]['solver'],
        n_discrete_t=hyperparameters[dataset]['n_discrete_t'],  # for training
        conditioned=conditioned,
        diffusion_lambda= 1,
        use_gan=False,
        gan_lambda= 1,
        device=device,
        rho=7,
        ema_rate=hyperparameters[dataset]['ema_rate'],
        n_sampling_steps=n_sampling_steps,  # for sampling/inference
        use_teacher=use_pretraining,
        datatype='image',
        num_classes=hyperparameters[dataset]['num_classes'],
    )

    # if not simultanous_training:
    # First pretrain the diffusion model and then train the consistency model
    if use_pretraining:
        for i in range(hyperparameters[dataset]['total_train_iters']):
            samples, cond = next(train_dataloader)
            samples = samples.to(device)
            cond = cond.reshape(-1, 1).to(device)  
            diff_loss = ctm.diffusion_train_step(samples, cond, i, hyperparameters[dataset]['total_train_iters'])
            pbar.set_description(f"Step {i}, Diff Loss: {diff_loss:.8f}")
            # break
        if eval_fid and i % eval_interval == 0:
            eval_model(ctm, dataset, image_shape, n_sampling_steps=n_sampling_steps, conditioned=conditioned, num_classes=hyperparameters[dataset]['num_classes'])

        
        ctm.update_teacher_model()
        
        plot_images(
            ctm, 
            image_shape,
            plot_n_samples,
            train_epochs, 
            sampling_method='euler', 
            n_sampling_steps=n_sampling_steps,
            save_path='./plots/',
            conditioned=conditioned,
            num_classes=hyperparameters[dataset]['num_classes'],
        )
    

    # Train the consistency trajectory model either simultanously with the diffusion model or after pretraining
    # data = iter(train_dataloader)
    for i in range(hyperparameters[dataset]['total_train_iters']):
        samples, cond = next(train_dataloader)
        samples = samples.to(device)
        cond = cond.reshape(-1, 1).to(device)  
        loss, ctm_loss, diffusion_loss, gan_loss = ctm.train_step(samples, cond, i, hyperparameters[dataset]['total_train_iters'])
        print(f"Step {i}, Loss: {loss:.4f}, CTM Loss: {ctm_loss:.4f}, Diff Loss: {diffusion_loss:.4f}, GAN Loss: {gan_loss:.4f}")
        # pbar.update(1)
        # break
        if eval_fid and i % eval_interval == 0:
            eval_model(ctm, dataset, image_shape, n_sampling_steps=n_sampling_steps, conditioned=conditioned, num_classes=hyperparameters[dataset]['num_classes'])

        if i % eval_interval == 0:   
            plot_images(
                ctm, 
                image_shape,
                plot_n_samples,
                i, 
                sampling_method='onestep', 
                n_sampling_steps=n_sampling_steps,
                save_path=plot_dir,
                conditioned=conditioned,
                num_classes=hyperparameters[dataset]['num_classes'],
            )

            plot_images(
                ctm, 
                image_shape,
                plot_n_samples,
                i, 
                sampling_method='multistep', 
                n_sampling_steps=n_sampling_steps,
                save_path=plot_dir,
                conditioned=conditioned,
                num_classes=hyperparameters[dataset]['num_classes'],
            )

            torch.save(ctm.state_dict(), f'ckpts/ctm_{dataset}.pth')

    # Plotting the results of the training
    # We do this for the one-step and the multi-step sampler to compare the results
    if not use_pretraining:
            plot_images(
                ctm, 
                image_shape,
                plot_n_samples,
                train_epochs, 
                sampling_method='euler', 
                n_sampling_steps=n_sampling_steps,
                save_path=plot_dir,
                conditioned=conditioned,
                num_classes=hyperparameters[dataset]['num_classes'],
            )

    plot_images(
        ctm, 
        image_shape,
        plot_n_samples,
        train_epochs, 
        sampling_method='onestep', 
        n_sampling_steps=n_sampling_steps,
        save_path=plot_dir,
        conditioned=conditioned,
        num_classes=hyperparameters[dataset]['num_classes'],
    )

    plot_images(
        ctm, 
        image_shape,
        plot_n_samples,
        train_epochs, 
        sampling_method='multistep', 
        n_sampling_steps=n_sampling_steps,
        save_path=plot_dir,
        conditioned=conditioned,
        num_classes=hyperparameters[dataset]['num_classes'],
    )
 
            
    print('done')