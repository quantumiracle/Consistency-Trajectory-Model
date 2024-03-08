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
import tensorflow.compat.v1 as tf
from ctm.evaluator import Evaluator
from ctm.visualization.vis_utils import sample_images

def calculate_inception_stats(image_path, detector_net, detector_kwargs, evaluator, feature_dim, data_name='cifar10', num_samples=50000, batch_size=100, device=th.device('cuda')):
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

    else:
        filenames = glob.glob(os.path.join(image_path, '*.npz'))
        imgs = []
        for file in filenames:
            try:
                img = np.load(file)  # ['arr_0']
                try:
                    img = img['data']
                except:
                    img = img['arr_0']
                imgs.append(img)
            except:
                pass
        imgs = np.concatenate(imgs, axis=0)
        os.makedirs(os.path.join(image_path, 'single_npz'), exist_ok=True)
        np.savez(os.path.join(os.path.join(image_path, 'single_npz'), f'data'),
                    imgs)  # , labels)
        print("computing sample batch activations...")
        sample_acts = evaluator.read_activations(
            os.path.join(os.path.join(image_path, 'single_npz'), f'data.npz'))
        print("computing/reading sample batch statistics...")
        sample_stats, sample_stats_spatial = tuple(evaluator.compute_statistics(x) for x in sample_acts)
        with open(os.path.join(os.path.join(image_path, 'single_npz'), f'stats'), 'wb') as f:
            pickle.dump({'stats': sample_stats, 'stats_spatial': sample_stats_spatial}, f)
        with open(os.path.join(os.path.join(image_path, 'single_npz'), f'acts'), 'wb') as f:
            pickle.dump({'acts': sample_acts[0], 'acts_spatial': sample_acts[1]}, f)
        return sample_acts, sample_stats, sample_stats_spatial


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


# https://github.com/sony/ctm/blob/main/code/cm/train_util.py#L1105
class Eval(object):
    def __init__(self, data_name='cifar10', delete=False, out=False):
        self.data_name = data_name
        self.delete = delete
        self.out = out

        print('Loading Inception-v3 model...')
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        self.detector_kwargs = dict(return_features=True)
        self.device  = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

        with dnnlib.util.open_url(detector_url, verbose=(0 == 0)) as f:
            self.detector_net = pickle.load(f).to(self.device)
        
        current_dir = os.getcwd() # Get the current working directory
        if data_name == 'cifar10':
            ref_path=os.path.join(current_dir, 'author_ckpt/cifar10_test.npz')
        else:
            ref_path=os.path.join(current_dir, 'author_ckpt/VIRTUAL_imagenet64_labeled.npz')

        print(ref_path)
        # with dnnlib.util.open_url(ref_path) as f:
        #     ref = dict(np.load(f))
        ref = dict(np.load(ref_path))
        self.mu_ref = ref['mu']
        self.sigma_ref = ref['sigma']

        config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.evaluator = Evaluator(tf.Session(config=config), batch_size=100)
        self.ref_acts = self.evaluator.read_activations(ref_path)
        self.ref_stats, self.ref_stats_spatial = self.evaluator.read_statistics(ref_path, self.ref_acts)

    def get_scores(self, model, dataset, image_shape, metric='fid', eval_num_samples=50000, n_sampling_steps=10, sample_dir='./plots/eval', conditioned=False, num_classes=10):
        sample_images(
            model,
            image_shape,
            eval_num_samples,
            sampling_method='euler', 
            n_sampling_steps=n_sampling_steps,
            save_path=sample_dir,
            conditioned=conditioned,
            num_classes=num_classes,
            )

        feature_dim = 2048 # for cifar10
        if self.data_name == 'cifar10':
            if metric == 'fid':
                mu, sigma = calculate_inception_stats(sample_dir, self.detector_net, self.detector_kwargs, self.evaluator, feature_dim,
                                                            num_samples=eval_num_samples, data_name=self.data_name, device=self.device)
                print(mu.shape, sigma.shape)
                # logger.log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate}"
                #             f" FID-{eval_num_samples // 1000}k: {compute_fid(mu, sigma, mu_ref=mu_ref, sigma_ref=sigma_ref)}")
                print(f"FID-{eval_num_samples // 1000}k: {compute_fid(mu, sigma, mu_ref=mu_ref, sigma_ref=sigma_ref)}")

            # if self.args.check_dm_performance:
            #     logger.log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate}"
            #                 f" FID-{eval_num_samples // 1000}k compared with DM: {self.compute_fid(mu, sigma, self.dm_mu, self.dm_sigma)}")
            #     self.calculate_similarity_metrics(sample_dir,
            #                                         num_samples=eval_num_samples)
            if self.delete:
                shutil.rmtree(sample_dir)
            if self.out:
                return compute_fid(mu, sigma)
            
        elif self.data_name == 'imagenet64':
            sample_acts, sample_stats, sample_stats_spatial = calculate_inception_stats(sample_dir, self.detector_net, self.detector_kwargs, self.evaluator, feature_dim, \
                                                            num_samples=eval_num_samples, data_name=self.data_name, device=self.device)
            print(f"Inception Score-{eval_num_samples // 1000}k:", self.evaluator.compute_inception_score(sample_acts[0]))
            print(f"FID-{eval_num_samples // 1000}k:", sample_stats.frechet_distance(self.ref_stats))
            print(f"sFID-{eval_num_samples // 1000}k:", sample_stats_spatial.frechet_distance(self.ref_stats_spatial))
            prec, recall = self.evaluator.compute_prec_recall(self.ref_acts[0], sample_acts[0])
            print("Precision:", prec)
            print("Recall:", recall)