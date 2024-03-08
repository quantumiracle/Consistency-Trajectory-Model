"""
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/image_datasets.py
"""

# from PIL import Image
# import blobfile as bf
# from mpi4py import MPI
# import numpy as np
# from torch.utils.data import DataLoader, Dataset


# def load_data(
#     *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
# ):
#     """
#     For a dataset, create a generator over (images, kwargs) pairs.

#     Each images is an NCHW float tensor, and the kwargs dict contains zero or
#     more keys, each of which map to a batched Tensor of their own.
#     The kwargs dict can be used for class labels, in which case the key is "y"
#     and the values are integer tensors of class labels.

#     :param data_dir: a dataset directory.
#     :param batch_size: the batch size of each returned pair.
#     :param image_size: the size to which images are resized.
#     :param class_cond: if True, include a "y" key in returned dicts for class
#                        label. If classes are not available and this is true, an
#                        exception will be raised.
#     :param deterministic: if True, yield results in a deterministic order.
#     """
#     if not data_dir:
#         raise ValueError("unspecified data directory")
#     all_files = _list_image_files_recursively(data_dir)
#     classes = None
#     if class_cond:
#         # Assume classes are the first part of the filename,
#         # before an underscore.
#         class_names = [bf.basename(path).split("_")[0] for path in all_files]
#         sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
#         classes = [sorted_classes[x] for x in class_names]
#     dataset = ImageDataset(
#         image_size,
#         all_files,
#         classes=classes,
#         shard=MPI.COMM_WORLD.Get_rank(),
#         num_shards=MPI.COMM_WORLD.Get_size(),
#     )
#     if deterministic:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
#         )
#     else:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
#         )
#     while True:
#         yield from loader


# def _list_image_files_recursively(data_dir):
#     results = []
#     for entry in sorted(bf.listdir(data_dir)):
#         full_path = bf.join(data_dir, entry)
#         ext = entry.split(".")[-1]
#         if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
#             results.append(full_path)
#         elif bf.isdir(full_path):
#             results.extend(_list_image_files_recursively(full_path))
#     return results


# class ImageDataset(Dataset):
#     def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
#         super().__init__()
#         self.resolution = resolution
#         self.local_images = image_paths[shard:][::num_shards]
#         self.local_classes = None if classes is None else classes[shard:][::num_shards]

#     def __len__(self):
#         return len(self.local_images)

#     def __getitem__(self, idx):
#         path = self.local_images[idx]
#         with bf.BlobFile(path, "rb") as f:
#             pil_image = Image.open(f)
#             pil_image.load()

#         # We are not on a new enough PIL to support the `reducing_gap`
#         # argument, which uses BOX downsampling at powers of two first.
#         # Thus, we do it by hand to improve downsample quality.
#         while min(*pil_image.size) >= 2 * self.resolution:
#             pil_image = pil_image.resize(
#                 tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#             )

#         scale = self.resolution / min(*pil_image.size)
#         pil_image = pil_image.resize(
#             tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#         )

#         arr = np.array(pil_image.convert("RGB"))
#         crop_y = (arr.shape[0] - self.resolution) // 2
#         crop_x = (arr.shape[1] - self.resolution) // 2
#         arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
#         arr = arr.astype(np.float32) / 127.5 - 1

#         out_dict = {}
#         if self.local_classes is not None:
#             out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
#         return np.transpose(arr, [2, 0, 1]), out_dict



"""
https://github.com/sony/ctm/blob/main/code/cm/image_datasets.py
"""

import math
import random

from PIL import Image
import blobfile as bf

import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from ctm import dist_util

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    data_name='cifar10',
    train_classes=None,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    num_workers=32,
    type='jpeg',
    flip_ratio=0.5,
    use_MPI=False,
    device_id = '1',
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    if train_classes == 281:
        data_dir = os.path.join(data_dir, 'n02123045')
        all_files = _list_image_files_recursively(data_dir, type=type)
    elif train_classes == -2:
        class_names = {'n01440764': 0, 'n01443537': 1, 'n01518878': 9, 'n01531178': 11, 'n01632777': 29, 'n01644373': 31,
                   'n01664065': 33, 'n01729977': 55, 'n01774750': 76, 'n01819313': 89, 'n01820546': 90, 'n02007558': 130,
                   'n02099601': 207, 'n02110185': 250, 'n02120079': 279, 'n02123045': 281, 'n02129165': 291, 'n02279972': 323,
                   'n02504458': 386, 'n02509815': 387, 'n02510455': 388, 'n02782093': 417, 'n03388043': 562, 'n03617480': 614,
                   'n04069434': 759, 'n04201297': 789, 'n04243546': 800, 'n04266014': 812, 'n04392985': 848, 'n07697313': 933,
                   'n09256479': 973, 'n09472597': 980}
        all_files = []
        classes = []
        for cl in list(class_names.keys()):
            data_dir_ = os.path.join(data_dir, cl)
            temp_files = _list_image_files_recursively(data_dir_, type=type)
            all_files.extend(temp_files)
            classes.extend([class_names[cl] for _ in temp_files])
    else:
        all_files = _list_image_files_recursively(data_dir, type=type)
        classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        if train_classes == 281:
            classes = [281 for _ in all_files]
        if train_classes not in [281, -2]:
            if data_name == 'cifar10':
                classes = [int(path.split('/')[-2]) for path in all_files]
            else:
                class_names = [bf.basename(path).split("_")[0] for path in all_files]
                sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
                classes = [sorted_classes[x] for x in class_names]

    if use_MPI:
        dist_util.setup_dist(device_id)
        from mpi4py import MPI
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            data_name=data_name,
            type=type,
            flip_ratio=flip_ratio,
        )
    else:
        # dist_util.setup_dist_without_MPI(device_id)
        # import torch.distributed as dist
        from mpi4py import MPI
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            # shard=dist.get_rank(),
            # num_shards=dist.get_world_size(),
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            data_name=data_name,
            type=type,
            flip_ratio=flip_ratio,
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir, type='jpeg'):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy", "jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        data_name='cifar10',
        type='jpeg',
        flip_ratio=0.5,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        print(f"Total number of data: {len(image_paths)}, data for {shard}/{num_shards} device: {len(self.local_images)}")
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.data_name = data_name
        self.type = type
        self.flip_ratio = flip_ratio

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        if self.type == 'npy':
            data = np.load(path)
            # out_dict = {}
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            # return data, out_dict
            """ use list instead of dict """
            out_list = np.array([])
            if self.local_classes is not None:
                out_list = np.array(self.local_classes[idx], dtype=np.int64)
            return data, out_list
                
        else:
            with bf.BlobFile(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()
            pil_image = pil_image.convert("RGB")

            if self.random_crop:
                arr = random_crop_arr(pil_image, self.resolution)
            else:
                arr = center_crop_arr(pil_image, self.resolution)

            if self.random_flip and random.random() < self.flip_ratio:
                arr = arr[:, ::-1]

            arr = arr.astype(np.float32) / 127.5 - 1

            # out_dict = {}
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            # return np.transpose(arr, [2, 0, 1]), out_dict
            """ use list instead of dict """
            out_list = np.array([])
            if self.local_classes is not None:
                out_list = np.array(self.local_classes[idx], dtype=np.int64)
            return np.transpose(arr, [2, 0, 1]), out_list

def center_crop_arr(pil_image, image_size, data_name='cifar10'):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    if data_name in ['church']:
        img = np.array(pil_image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if image_size is not None:
            image = image.resize((image_size, image_size), resample='bicubic')
        return image
    else:
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]