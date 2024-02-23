import torch as th
import glob
import os
import cv2
import scipy
import dnnlib
import pickle
import shutil
import numpy as np
from skimage.metrics import structural_similarity as SSIM_

def calculate_inception_stats(image_path, detector_net, detector_kwargs, feature_dim, data_name='cifar10', num_samples=50000, batch_size=100, device=th.device('cuda')):
    if data_name.lower() == 'cifar10':
        print(f'Loading images from "{image_path}"...')
        mu = th.zeros([feature_dim], dtype=th.float64, device=device)
        sigma = th.zeros([feature_dim, feature_dim], dtype=th.float64, device=device)
        files = glob.glob(os.path.join(image_path, 'sample*.npz'))
        count = 0
        for file in files:
            images = np.load(file)['arr_0']  # [0]#["samples"]
            for k in range((images.shape[0] - 1) // batch_size + 1):
                mic_img = images[k * batch_size: (k + 1) * batch_size]
                mic_img = th.tensor(mic_img).permute(0, 3, 1, 2).to(device)
                features = detector_net(mic_img, **detector_kwargs).to(th.float64)
                if count + mic_img.shape[0] > num_samples:
                    remaining_num_samples = num_samples - count
                else:
                    remaining_num_samples = mic_img.shape[0]
                mu += features[:remaining_num_samples].sum(0)
                sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
                count = count + remaining_num_samples
                print(count)
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
        assert count == num_samples
        print(count)
        mu /= num_samples
        sigma -= mu.ger(mu) * num_samples
        sigma /= num_samples - 1
        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()
        return mu, sigma

# def calculate_similarity_metrics(image_path, num_samples=50000):
#     files = glob.glob(os.path.join(image_path, 'sample*.npz'))
#     files.sort()
#     count = 0
#     psnr = 0
#     ssim = 0
#     for i, file in enumerate(files):
#         images = np.load(file)['arr_0']
#         for k in range((images.shape[0] - 1) // batch_size + 1):
#             #ref_img = self.ref_images[count + k * batch_size: count + (k + 1) * batch_size]
#             if count + batch_size > num_samples:
#                 remaining_num_samples = num_samples - count
#             else:
#                 remaining_num_samples = batch_size
#             img = images[k * batch_size: k * batch_size + remaining_num_samples]
#             ref_img = ref_images[count: count + remaining_num_samples]
#             psnr += cv2.PSNR(img, ref_img) * remaining_num_samples
#             ssim += SSIM_(img,ref_img,multichannel=True,channel_axis=3,data_range=255) * remaining_num_samples
#             count = count + remaining_num_samples
#             print(count)
#             if count >= num_samples:
#                 break
#         if count >= num_samples:
#             break
#     assert count == num_samples
#     print(count)
#     psnr /= num_samples
#     ssim /= num_samples
#     assert num_samples % 1000 == 0
#     return psnr, ssim

def compute_fid(mu, sigma, ref_mu=None, ref_sigma=None, mu_ref=None, sigma_ref=None):
    if np.array(ref_mu == None).sum():
        ref_mu = mu_ref
        assert ref_sigma == None
        ref_sigma = sigma_ref
    m = np.square(mu - ref_mu).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, ref_sigma), disp=False)
    fid = m + np.trace(sigma + ref_sigma - s * 2)
    fid = float(np.real(fid))
    return fid


def eval(sample_dir, data_name='cifar10', metric='fid', eval_num_samples=50000, delete=False, out=False):
    print('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    device  = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

    with dnnlib.util.open_url(detector_url, verbose=(0 == 0)) as f:
        detector_net = pickle.load(f).to(device)
    
    current_dir = os.getcwd() # Get the current working directory
    ref_path=os.path.join(current_dir, 'author_ckpt/cifar10_test.npz')
    print(ref_path)
    # with dnnlib.util.open_url(ref_path) as f:
    #     ref = dict(np.load(f))
    ref = dict(np.load(ref_path))
    mu_ref = ref['mu']
    sigma_ref = ref['sigma']

    feature_dim = 2048 # for cifar10

    if data_name == 'cifar10':
        if metric == 'fid':
            mu, sigma = calculate_inception_stats(sample_dir, detector_net, detector_kwargs, feature_dim,
                                                        num_samples=eval_num_samples, data_name=data_name, device=device)
            print(mu.shape, sigma.shape)
            # logger.log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate}"
            #             f" FID-{eval_num_samples // 1000}k: {compute_fid(mu, sigma, mu_ref=mu_ref, sigma_ref=sigma_ref)}")
            print(f"FID-{eval_num_samples // 1000}k: {compute_fid(mu, sigma, mu_ref=mu_ref, sigma_ref=sigma_ref)}")

        # if self.args.check_dm_performance:
        #     logger.log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate}"
        #                 f" FID-{eval_num_samples // 1000}k compared with DM: {self.compute_fid(mu, sigma, self.dm_mu, self.dm_sigma)}")
        #     self.calculate_similarity_metrics(sample_dir,
        #                                         num_samples=eval_num_samples)
        if delete:
            shutil.rmtree(sample_dir)
        if out:
            return compute_fid(mu, sigma)