
from pathlib import Path
import os
import random
import json
import itertools
import copy

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, \
    SequentialSampler
from torchvision import transforms
import numpy as np
import cv2
import PIL
import scipy.io
import glob

from . import utils

default_data_dir = Path(__file__).resolve().parent.parent / "data"

# Set default paths
if "DReye" not in os.environ:
    os.environ["DReye_DATA_DIR"] = str(default_data_dir / "New_DReye")

if "DADA2000_DATA_DIR" not in os.environ:
    os.environ["DADA2000_DATA_DIR"] = str(default_data_dir / "DADA")

if "DT16_DATA_DIR" not in os.environ:
    os.environ["DT16_DATA_DIR"] = str(default_data_dir / "DT16")

if "BDDA_DATA_DIR" not in os.environ:
    os.environ["BDDA_DATA_DIR"] = str(default_data_dir / "BDDA")

config_path = Path(__file__).resolve().parent / "cache"

# os.environ["DADA2000_DATA_DIR"] = "/media/acl/7A4A85A74A85612D/01_Driver_Gaze/TASED_Net_DADA/data"


def get_dataloader(src='DHF1K'):
    if src in ('MIT1003',):
        return ImgSizeDataLoader
    return DataLoader


class ImgSizeBatchSampler:

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        assert(isinstance(dataset, MIT1003Dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        out_size_array = [
            dataset.size_dict[img_idx]['out_size']
            for img_idx in dataset.samples]
        self.out_size_set = sorted(list(set(out_size_array)))
        self.sample_idx_dict = {
            out_size: [] for out_size in self.out_size_set}
        for sample_idx, img_idx in enumerate(dataset.samples):
            self.sample_idx_dict[dataset.size_dict[img_idx]['out_size']].append(
                sample_idx)

        self.len = 0
        self.n_batches_dict = {}
        for out_size, sample_idx_array in self.sample_idx_dict.items():
            this_n_batches = len(sample_idx_array) // self.batch_size
            self.len += this_n_batches
            self.n_batches_dict[out_size] = this_n_batches

    def __iter__(self):
        batch_array = list(itertools.chain.from_iterable(
            [out_size for _ in range(n_batches)]
            for out_size, n_batches in self.n_batches_dict.items()))
        if not self.shuffle:
            random.seed(27)
        random.shuffle(batch_array)

        this_sample_idx_dict = copy.deepcopy(self.sample_idx_dict)
        for sample_idx_array in this_sample_idx_dict.values():
            random.shuffle(sample_idx_array)
        for out_size in batch_array:
            this_indices = this_sample_idx_dict[out_size][:self.batch_size]
            del this_sample_idx_dict[out_size][:self.batch_size]
            yield this_indices

    def __len__(self):
        return self.len


class ImgSizeDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False,
                 **kwargs):
        if batch_size == 1:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        else:
            batch_sampler = ImgSizeBatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle,
                drop_last=drop_last)
        super().__init__(dataset, batch_sampler=batch_sampler, **kwargs)

def get_optimal_out_size(img_size):
    ar = img_size[0] / img_size[1]
    min_prod = 100
    max_prod = 120
    ar_array = []
    size_array = []
    for n1 in range(7, 14):
        for n2 in range(7, 14):
            if min_prod <= n1 * n2 <= max_prod:
                this_ar = n1 / n2
                this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
                ar_array.append(this_ar_ratio)
                size_array.append((n1, n2))

    max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
    bn_size = size_array[max_ar_ratio_idx]
    out_size = tuple(r * 32 for r in bn_size)
    return out_size


class FolderVideoDataset(Dataset):

    def __init__(self, images_path, frame_modulo=None, source=None):
        self.images_path = images_path
        self.frame_modulo = frame_modulo or 5
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }

        frame_files = sorted(list(images_path.glob("*")))
        frame_files = [file for file in frame_files
                         if file.suffix in ('.png', '.jpg', '.jpeg')]
        self.frame_files = frame_files
        self.vid_nr_array = [0]
        self.n_images_dict = {0: len(frame_files)}

        img = cv2.imread(str(frame_files[0]))
        img_size = tuple(img.shape[:2])
        self.target_size_dict = {0: img_size}

        if source == 'DHF1K' and img_size == (360, 640):
            self.out_size = (224, 384)

        elif source == 'Hollywood':
            self.out_size = (224, 416)

        elif source == 'UCFSports':
            self.out_size = (256, 384)

        else:
            self.out_size = get_optimal_out_size(img_size)

    def load_frame(self, f_nr):
        frame_file = self.frame_files[f_nr - 1]
        frame = cv2.imread(str(frame_file))
        if frame is None:
            raise FileNotFoundError(frame_file)
        frame = np.ascontiguousarray(frame[:, :, ::-1])
        return frame

    def preprocess_sequence(self, frame_seq):
        transformations = []
        transformations.append(transforms.ToPILImage())
        transformations.append(transforms.Resize(
            self.out_size, interpolation=PIL.Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        processing = transforms.Compose(transformations)
        tensor = [processing(img) for img in frame_seq]
        tensor = torch.stack(tensor)
        return tensor

    def get_data(self, vid_nr, start):
        n_images = self.n_images_dict[vid_nr]
        frame_nrs = list(range(start, n_images + 1, self.frame_modulo))
        frame_seq = [self.load_frame(f_nr) for f_nr in frame_nrs]
        frame_seq = self.preprocess_sequence(frame_seq)
        target_size = self.target_size_dict[vid_nr]
        return frame_nrs, frame_seq, target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.get_data(item, 0)


class FolderImageDataset(Dataset):

    def __init__(self, images_path):
        self.images_path = images_path
        self.frame_modulo = 1
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }

        image_files = sorted(list(images_path.glob("*")))
        image_files = [file for file in image_files
                       if file.suffix in ('.png', '.jpg', '.jpeg')]
        self.image_files = image_files
        self.n_images_dict = {
            img_idx: 1 for img_idx in range(len(self.image_files))}

        self.target_size_dict = {}
        self.out_size_dict = {}
        for img_idx, file in enumerate(image_files):
            img = cv2.imread(str(file))
            img_size = tuple(img.shape[:2])
            self.target_size_dict[img_idx] = img_size
            self.out_size_dict[img_idx] = get_optimal_out_size(img_size)

    def load_image(self, img_idx):
        image_file = self.image_files[img_idx]
        image = cv2.imread(str(image_file))
        if image is None:
            raise FileNotFoundError(image_file)
        image = np.ascontiguousarray(image[:, :, ::-1])
        return image

    def preprocess(self, img, out_size):
        transformations = [
            transforms.ToPILImage(),
            transforms.Resize(
                out_size, interpolation=PIL.Image.LANCZOS),
            transforms.ToTensor(),
        ]
        if 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor

    def get_data(self, img_idx):
        file = self.image_files[img_idx]
        img = cv2.imread(str(file))
        assert (img is not None)
        img = np.ascontiguousarray(img[:, :, ::-1])
        out_size = self.out_size_dict[img_idx]
        img = self.preprocess(img, out_size)
        return [1], img, self.target_size_dict[img_idx]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        return self.get_data(item, 0)

###

class DReyeDataset(Dataset, utils.KwConfigClass):

    img_channels = 1
    n_train_val_videos = 405 # 570
    test_vid_nrs = (406, 780)  #1110
    frame_rate = 24 # note video 25fps and modify frame_modulo=4
    source = 'DReye'
    dynamic = True

    def __init__(self,
                 seq_len=12,
                 frame_modulo=4,
                 max_seq_len=1e6,
                 preproc_cfg=None,
                 out_size=(224, 384), phase='train', target_size=(360, 640),
                 debug=False, val_size=27, n_x_val=3, x_val_step=2,
                 x_val_seed=0, seq_per_vid=1, subset=None, verbose=1,
                 n_images_file='DReye_n_images.dat', seq_per_vid_val=2,
                 sal_offset=None):
        self.phase = phase
        self.train = phase == 'train'
        if not self.train:
            preproc_cfg = {}
        elif preproc_cfg is None:
            preproc_cfg = {}
        preproc_cfg.update({
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        })
        self.preproc_cfg = preproc_cfg
        self.out_size = out_size
        self.debug = debug
        self.val_size = val_size
        self.n_x_val = n_x_val
        self.x_val_step = x_val_step
        self.x_val_seed = x_val_seed
        self.seq_len = seq_len
        self.seq_per_vid = seq_per_vid
        self.seq_per_vid_val = seq_per_vid_val
        self.frame_modulo = frame_modulo
        self.clip_len = seq_len * frame_modulo
        self.subset = subset
        self.verbose = verbose
        self.n_images_file = n_images_file
        self.target_size = target_size
        self.sal_offset = sal_offset
        self.max_seq_len = max_seq_len
        self._dir = None
        self._n_images_dict = None
        self.vid_nr_array = None

        # Evaluation
        if phase in ('eval', 'test'):
            self.seq_len = int(1e6)

        if self.phase in ('test',):
            self.vid_nr_array = list(range(
                self.test_vid_nrs[0], self.test_vid_nrs[1] + 1))
            self.samples, self.target_size_dict = self.prepare_samples()
            return

        # Cross-validation split
        n_videos = self.n_train_val_videos
        assert(self.val_size <= n_videos // self.n_x_val)
        assert(self.x_val_step < self.n_x_val)
        vid_nr_array = np.arange(1, n_videos + 1)
        if self.x_val_seed > 0:
            np.random.seed(self.x_val_seed)
            np.random.shuffle(vid_nr_array)
        val_start = (len(vid_nr_array) - self.val_size) //\
                    (self.n_x_val - 1) * self.x_val_step
        vid_nr_array = vid_nr_array.tolist()
        if not self.train:
            self.vid_nr_array =\
                vid_nr_array[val_start:val_start + self.val_size]
        else:
            del vid_nr_array[val_start:val_start + self.val_size]
            self.vid_nr_array = vid_nr_array

        if self.subset is not None:
            self.vid_nr_array =\
                self.vid_nr_array[:int(len(self.vid_nr_array) * self.subset)]

        self.samples, self.target_size_dict = self.prepare_samples()

    @property
    def n_images_dict(self):
        if self._n_images_dict is None:
            with open(config_path.parent / self.n_images_file, 'r') as f:
                self._n_images_dict = {
                    idx + 1: int(line) for idx, line in enumerate(f)
                    if idx + 1 in self.vid_nr_array}
        return self._n_images_dict

    @property
    def dir(self):
        if self._dir is None:
            self._dir = Path(os.environ["DReye_DATA_DIR"])
        return self._dir

    @property
    def n_samples(self):
        return len(self.vid_nr_array)

    def __len__(self):
        return len(self.samples)

    def prepare_samples(self):
        samples = []
        too_short = 0
        too_long = 0
        for vid_nr, n_images in self.n_images_dict.items():
            if self.phase in ('eval', 'test'):
                samples += [
                    (vid_nr, offset + 1) for offset in range(self.frame_modulo)]
                continue
            # 帧数过小多大直接跳过
            if n_images < self.clip_len:
                too_short += 1
                continue
            if n_images // self.frame_modulo > self.max_seq_len:
                too_long += 1
                continue
            # 
            if self.phase == 'train':
                samples += [(vid_nr, None)] * self.seq_per_vid
                continue
            elif self.phase == 'valid':
                x = n_images // (self.seq_per_vid_val * 2) - self.clip_len // 2
                start = max(1, x)
                end = min(n_images - self.clip_len, n_images - x)
                samples += [
                    (vid_nr, int(start)) for start in
                    np.linspace(start, end, self.seq_per_vid_val)]
                continue
        # 打印数据集加载的基本信息
        if self.phase not in ('eval', 'test') and self.n_images_dict:
            n_loaded = len(self.n_images_dict) - too_short - too_long
            print(f"{n_loaded} videos loaded "
                  f"({n_loaded / len(self.n_images_dict) * 100:.1f}%)")
            print(f"{too_short} videos are too short "
                  f"({too_short / len(self.n_images_dict) * 100:.1f}%)")
            print(f"{too_long} videos are too long "
                  f"({too_long / len(self.n_images_dict) * 100:.1f}%)")
        target_size_dict = {
            vid_nr: self.target_size for vid_nr in self.n_images_dict.keys()}
        return samples, target_size_dict

    def get_frame_nrs(self, vid_nr, start):
        n_images = self.n_images_dict[vid_nr]
        if self.phase in ('eval', 'test'):
            return list(range(start, n_images + 1, self.frame_modulo))
        return list(range(start, start + self.clip_len, self.frame_modulo))

    def get_data_file(self, vid_nr, f_nr, dkey):
        if dkey == 'frame':
            folder = 'images'
        elif dkey == 'sal':
            folder = 'new_maps'
        elif dkey == 'fix':
            folder = 'fixation'
        else:
            raise ValueError(f'Unknown data key {dkey}')
        ###
        img_path = str(self.dir  / f'{vid_nr:04d}' / folder/ f'{f_nr:04d}.png')
        return img_path

    def load_data(self, vid_nr, f_nr, dkey):
        read_flag = None if dkey == 'frame' else cv2.IMREAD_GRAYSCALE
        data_file = self.get_data_file(vid_nr, f_nr, dkey)
        if read_flag is not None:
            data = cv2.imread(str(data_file), read_flag)
        else:
            data = cv2.imread(str(data_file))
        if data is None:
            raise FileNotFoundError(data_file)
        if dkey == 'frame':
            data = np.ascontiguousarray(data[:, :, ::-1])

        if dkey == 'sal' and self.train and self.sal_offset is not None:
            data += self.sal_offset
            data[0, 0] = 0

        return data

    def preprocess_sequence(self, frame_seq, dkey, vid_nr):
        transformations = []
        if dkey == 'frame':
            transformations.append(transforms.ToPILImage())
            transformations.append(transforms.Resize(
                self.out_size, interpolation=PIL.Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if dkey == 'frame' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif dkey == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        # elif dkey == 'fix':
        #     transformations.append(
        #         transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))
        ##！
        processing = transforms.Compose(transformations)

        tensor = [processing(img) for img in frame_seq]
        tensor = torch.stack(tensor)
        return tensor

    def get_seq(self, vid_nr, frame_nrs, dkey):
        data_seq = [self.load_data(vid_nr, f_nr, dkey) for f_nr in frame_nrs]
        return self.preprocess_sequence(data_seq, dkey, vid_nr)

    def get_data(self, vid_nr, start):
        if start is None:
            max_start = self.n_images_dict[vid_nr] - self.clip_len + 1
            if max_start == 1:
                start = max_start
            else:
                start = np.random.randint(1, max_start)
        frame_nrs = self.get_frame_nrs(vid_nr, start)
        frame_seq = self.get_seq(vid_nr, frame_nrs, 'frame')
        target_size = self.target_size_dict[vid_nr]
        # if self.phase == 'test' and self.source in ('DReye',):
        #     return frame_nrs, frame_seq, target_size
        sal_seq = self.get_seq(vid_nr, frame_nrs, 'sal')

        fix_seq = torch.full(self.target_size, 0, dtype=torch.bool)
        # fix used for nss aucj and aucs
        # fix_seq = self.get_seq(vid_nr, frame_nrs, 'fix')
        # 用 sal_seq替换fix_seq
        return frame_nrs, frame_seq, sal_seq, fix_seq, target_size

    def __getitem__(self, item):
        vid_nr, start = self.samples[item]
        data = self.get_data(vid_nr, start)
        return data


class DADA2000Dataset(Dataset, utils.KwConfigClass):

    img_channels = 1
    n_train_val_videos = 797
    test_vid_nrs = (798, 1013)
    frame_rate = 30
    source = 'DADA200'
    dynamic = True

    def __init__(self,
                 seq_len=12,
                 frame_modulo=5,
                 max_seq_len=1e6,
                 preproc_cfg=None,
                 out_size=(224, 538), phase='train', target_size=(224, 538),
                 debug=False, val_size=100, n_x_val=3, x_val_step=2,
                 x_val_seed=0, seq_per_vid=1, subset=None, verbose=1,
                 n_images_file='DADA_n_images.dat', seq_per_vid_val=2,
                 sal_offset=None):
        self.phase = phase
        self.train = phase == 'train'
        if not self.train:
            preproc_cfg = {}
        elif preproc_cfg is None:
            preproc_cfg = {}
        preproc_cfg.update({
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        })
        self.preproc_cfg = preproc_cfg
        self.out_size = out_size
        self.debug = debug
        self.val_size = val_size
        self.n_x_val = n_x_val
        self.x_val_step = x_val_step
        self.x_val_seed = x_val_seed
        self.seq_len = seq_len
        self.seq_per_vid = seq_per_vid
        self.seq_per_vid_val = seq_per_vid_val
        self.frame_modulo = frame_modulo
        self.clip_len = seq_len * frame_modulo
        self.subset = subset
        self.verbose = verbose
        self.n_images_file = n_images_file
        self.target_size = target_size
        self.sal_offset = sal_offset
        self.max_seq_len = max_seq_len
        self._dir = None
        self._n_images_dict = None
        self.vid_nr_array = None

        # Evaluation
        if phase in ('eval', 'test'):
            self.seq_len = int(1e6)

        if self.phase in ('test',):
            self.vid_nr_array = list(range(
                self.test_vid_nrs[0], self.test_vid_nrs[1] + 1))
            self.samples, self.target_size_dict = self.prepare_samples()
            return

        # Cross-validation split
        n_videos = self.n_train_val_videos
        assert(self.val_size <= n_videos // self.n_x_val)
        assert(self.x_val_step < self.n_x_val)
        vid_nr_array = np.arange(1, n_videos + 1)
        if self.x_val_seed > 0:
            np.random.seed(self.x_val_seed)
            np.random.shuffle(vid_nr_array)
        val_start = (len(vid_nr_array) - self.val_size) //\
                    (self.n_x_val - 1) * self.x_val_step
        vid_nr_array = vid_nr_array.tolist()
        if not self.train:
            self.vid_nr_array =\
                vid_nr_array[val_start:val_start + self.val_size]
        else:
            del vid_nr_array[val_start:val_start + self.val_size]
            self.vid_nr_array = vid_nr_array

        if self.subset is not None:
            self.vid_nr_array =\
                self.vid_nr_array[:int(len(self.vid_nr_array) * self.subset)]

        self.samples, self.target_size_dict = self.prepare_samples()

    @property
    def n_images_dict(self):
        if self._n_images_dict is None:
            with open(config_path.parent / self.n_images_file, 'r') as f:
                self._n_images_dict = {
                    idx + 1: int(line) for idx, line in enumerate(f)
                    if idx + 1 in self.vid_nr_array}
        return self._n_images_dict

    @property
    def dir(self):
        if self._dir is None:
            self._dir = Path(os.environ["DADA2000_DATA_DIR"])
        return self._dir

    @property
    def n_samples(self):
        return len(self.vid_nr_array)

    def __len__(self):
        return len(self.samples)

    def prepare_samples(self):
        samples = []
        too_short = 0
        too_long = 0
        for vid_nr, n_images in self.n_images_dict.items():
            if self.phase in ('eval', 'test'):
                samples += [
                    (vid_nr, offset + 1) for offset in range(self.frame_modulo)]
                continue
            # 帧数过小多大直接跳过
            if n_images < self.clip_len:
                too_short += 1
                continue
            if n_images // self.frame_modulo > self.max_seq_len:
                too_long += 1
                continue
            # 
            if self.phase == 'train':
                samples += [(vid_nr, None)] * self.seq_per_vid
                continue
            elif self.phase == 'valid':
                x = n_images // (self.seq_per_vid_val * 2) - self.clip_len // 2
                start = max(1, x)
                end = min(n_images - self.clip_len, n_images - x)
                samples += [
                    (vid_nr, int(start)) for start in
                    np.linspace(start, end, self.seq_per_vid_val)]
                continue
        # 打印数据集加载的基本信息
        if self.phase not in ('eval', 'test') and self.n_images_dict:
            n_loaded = len(self.n_images_dict) - too_short - too_long
            print(f"{n_loaded} videos loaded "
                  f"({n_loaded / len(self.n_images_dict) * 100:.1f}%)")
            print(f"{too_short} videos are too short "
                  f"({too_short / len(self.n_images_dict) * 100:.1f}%)")
            print(f"{too_long} videos are too long "
                  f"({too_long / len(self.n_images_dict) * 100:.1f}%)")
        target_size_dict = {
            vid_nr: self.target_size for vid_nr in self.n_images_dict.keys()}
        return samples, target_size_dict

    def get_frame_nrs(self, vid_nr, start):
        n_images = self.n_images_dict[vid_nr]
        if self.phase in ('eval', 'test'):
            return list(range(start, n_images + 1, self.frame_modulo))
        return list(range(start, start + self.clip_len, self.frame_modulo))

    def get_data_file(self, vid_nr, f_nr, dkey):
        if dkey == 'frame':
            folder = 'images'
        elif dkey == 'sal':
            folder = 'maps'
        elif dkey == 'fix':
            folder = 'fixation'
        else:
            raise ValueError(f'Unknown data key {dkey}')
        ###
        img_path = str(self.dir  / f'{vid_nr:04d}' / folder/ f'{f_nr:04d}.png')
        return img_path

    def load_data(self, vid_nr, f_nr, dkey):
        read_flag = None if dkey == 'frame' else cv2.IMREAD_GRAYSCALE
        data_file = self.get_data_file(vid_nr, f_nr, dkey)
        if read_flag is not None:
            data = cv2.imread(str(data_file), read_flag)
        else:
            data = cv2.imread(str(data_file))
        if data is None:
            raise FileNotFoundError(data_file)
        if dkey == 'frame':
            data = np.ascontiguousarray(data[:, :, ::-1])

        if dkey == 'sal' and self.train and self.sal_offset is not None:
            data += self.sal_offset
            data[0, 0] = 0

        return data

    def preprocess_sequence(self, frame_seq, dkey, vid_nr):
        transformations = []
        if dkey == 'frame':
            transformations.append(transforms.ToPILImage())
            transformations.append(transforms.Resize(
                self.out_size, interpolation=PIL.Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if dkey == 'frame' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif dkey == 'sal':
            transformations.append(transforms.ToPILImage())
            transformations.append(transforms.Resize(
                self.out_size, interpolation=PIL.Image.LANCZOS))
            transformations.append(transforms.ToTensor())
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        # elif dkey == 'fix':
        #     transformations.append(
        #         transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))
        ##！
        processing = transforms.Compose(transformations)

        tensor = [processing(img) for img in frame_seq]
        tensor = torch.stack(tensor)
        return tensor

    def get_seq(self, vid_nr, frame_nrs, dkey):
        data_seq = [self.load_data(vid_nr, f_nr, dkey) for f_nr in frame_nrs]
        return self.preprocess_sequence(data_seq, dkey, vid_nr)

    def get_data(self, vid_nr, start):
        if start is None:
            max_start = self.n_images_dict[vid_nr] - self.clip_len + 1
            if max_start == 1:
                start = max_start
            else:
                start = np.random.randint(1, max_start)
        frame_nrs = self.get_frame_nrs(vid_nr, start)
        frame_seq = self.get_seq(vid_nr, frame_nrs, 'frame')
        target_size = self.target_size_dict[vid_nr]
        # if self.phase == 'test' and self.source in ('DADA2000',):
        #     return frame_nrs, frame_seq, target_size
        sal_seq = self.get_seq(vid_nr, frame_nrs, 'sal')

        fix_seq = torch.full(self.target_size, 0, dtype=torch.bool)
        # fix used for nss aucj and aucs
        # fix_seq = self.get_seq(vid_nr, frame_nrs, 'fix')
        # 用 sal_seq替换fix_seq
        return frame_nrs, frame_seq, sal_seq, fix_seq, target_size

    def __getitem__(self, item):
        vid_nr, start = self.samples[item]
        data = self.get_data(vid_nr, start)
        return data


class DT16Dataset(Dataset, utils.KwConfigClass):

    img_channels = 1
    n_train_val_videos = 115
    test_vid_nrs = (115, 153)  #1110
    frame_rate = 24 
    source = 'DT16'
    dynamic = True

    def __init__(self,
                 seq_len=12,
                 frame_modulo=4,
                 max_seq_len=1e6,
                 preproc_cfg=None,
                 out_size=(224, 384), phase='train', target_size=(360, 640),
                 debug=False, val_size=19, n_x_val=3, x_val_step=2,
                 x_val_seed=0, seq_per_vid=1, subset=None, verbose=1,
                 n_images_file='DT16_n_images.dat', seq_per_vid_val=2,
                 sal_offset=None):
        self.phase = phase
        self.train = phase == 'train'
        if not self.train:
            preproc_cfg = {}
        elif preproc_cfg is None:
            preproc_cfg = {}
        preproc_cfg.update({
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        })
        self.preproc_cfg = preproc_cfg
        self.out_size = out_size
        self.debug = debug
        self.val_size = val_size
        self.n_x_val = n_x_val
        self.x_val_step = x_val_step
        self.x_val_seed = x_val_seed
        self.seq_len = seq_len
        self.seq_per_vid = seq_per_vid
        self.seq_per_vid_val = seq_per_vid_val
        self.frame_modulo = frame_modulo
        self.clip_len = seq_len * frame_modulo
        self.subset = subset
        self.verbose = verbose
        self.n_images_file = n_images_file
        self.target_size = target_size
        self.sal_offset = sal_offset
        self.max_seq_len = max_seq_len
        self._dir = None
        self._n_images_dict = None
        self.vid_nr_array = None

        # Evaluation
        if phase in ('eval', 'test'):
            self.seq_len = int(1e6)

        if self.phase in ('test',):
            self.vid_nr_array = list(range(
                self.test_vid_nrs[0], self.test_vid_nrs[1] + 1))
            self.samples, self.target_size_dict = self.prepare_samples()
            return

        # Cross-validation split
        n_videos = self.n_train_val_videos
        assert(self.val_size <= n_videos // self.n_x_val)
        assert(self.x_val_step < self.n_x_val)
        vid_nr_array = np.arange(1, n_videos + 1)
        if self.x_val_seed > 0:
            np.random.seed(self.x_val_seed)
            np.random.shuffle(vid_nr_array)
        val_start = (len(vid_nr_array) - self.val_size) //\
                    (self.n_x_val - 1) * self.x_val_step
        vid_nr_array = vid_nr_array.tolist()
        if not self.train:
            self.vid_nr_array =\
                vid_nr_array[val_start:val_start + self.val_size]
        else:
            del vid_nr_array[val_start:val_start + self.val_size]
            self.vid_nr_array = vid_nr_array

        if self.subset is not None:
            self.vid_nr_array =\
                self.vid_nr_array[:int(len(self.vid_nr_array) * self.subset)]

        self.samples, self.target_size_dict = self.prepare_samples()

    @property
    def n_images_dict(self):
        if self._n_images_dict is None:
            with open(config_path.parent / self.n_images_file, 'r') as f:
                self._n_images_dict = {
                    idx + 1: int(line) for idx, line in enumerate(f)
                    if idx + 1 in self.vid_nr_array}
        return self._n_images_dict

    @property
    def dir(self):
        if self._dir is None:
            self._dir = Path(os.environ["DT16_DATA_DIR"])
        return self._dir

    @property
    def n_samples(self):
        return len(self.vid_nr_array)

    def __len__(self):
        return len(self.samples)

    def prepare_samples(self):
        samples = []
        too_short = 0
        too_long = 0
        for vid_nr, n_images in self.n_images_dict.items():
            if self.phase in ('eval', 'test'):
                samples += [
                    (vid_nr, offset + 1) for offset in range(self.frame_modulo)]
                continue
            # 帧数过小多大直接跳过
            if n_images < self.clip_len:
                too_short += 1
                continue
            if n_images // self.frame_modulo > self.max_seq_len:
                too_long += 1
                continue
            # 
            if self.phase == 'train':
                samples += [(vid_nr, None)] * self.seq_per_vid
                continue
            elif self.phase == 'valid':
                x = n_images // (self.seq_per_vid_val * 2) - self.clip_len // 2
                start = max(1, x)
                end = min(n_images - self.clip_len, n_images - x)
                samples += [
                    (vid_nr, int(start)) for start in
                    np.linspace(start, end, self.seq_per_vid_val)]
                continue
        # 打印数据集加载的基本信息
        if self.phase not in ('eval', 'test') and self.n_images_dict:
            n_loaded = len(self.n_images_dict) - too_short - too_long
            print(f"{n_loaded} videos loaded "
                  f"({n_loaded / len(self.n_images_dict) * 100:.1f}%)")
            print(f"{too_short} videos are too short "
                  f"({too_short / len(self.n_images_dict) * 100:.1f}%)")
            print(f"{too_long} videos are too long "
                  f"({too_long / len(self.n_images_dict) * 100:.1f}%)")
        target_size_dict = {
            vid_nr: self.target_size for vid_nr in self.n_images_dict.keys()}
        return samples, target_size_dict

    def get_frame_nrs(self, vid_nr, start):
        n_images = self.n_images_dict[vid_nr]
        if self.phase in ('eval', 'test'):
            return list(range(start, n_images + 1, self.frame_modulo))
        return list(range(start, start + self.clip_len, self.frame_modulo))

    def get_data_file(self, vid_nr, f_nr, dkey):
        if dkey == 'frame':
            folder = 'images'
        elif dkey == 'sal':
            folder = 'maps'
        elif dkey == 'fix':
            folder = 'fixation'
        else:
            raise ValueError(f'Unknown data key {dkey}')
        ###
        img_path = str(self.dir  / f'{vid_nr:04d}' / folder/ f'{f_nr:04d}.png')
        return img_path

    def load_data(self, vid_nr, f_nr, dkey):
        read_flag = None if dkey == 'frame' else cv2.IMREAD_GRAYSCALE
        data_file = self.get_data_file(vid_nr, f_nr, dkey)
        if read_flag is not None:
            data = cv2.imread(str(data_file), read_flag)
        else:
            data = cv2.imread(str(data_file))
        if data is None:
            raise FileNotFoundError(data_file)
        if dkey == 'frame':
            data = np.ascontiguousarray(data[:, :, ::-1])

        if dkey == 'sal' and self.train and self.sal_offset is not None:
            data += self.sal_offset
            data[0, 0] = 0

        return data

    def preprocess_sequence(self, frame_seq, dkey, vid_nr):
        transformations = []
        if dkey == 'frame':
            transformations.append(transforms.ToPILImage())
            transformations.append(transforms.Resize(
                self.out_size, interpolation=PIL.Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if dkey == 'frame' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif dkey == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        # elif dkey == 'fix':
        #     transformations.append(
        #         transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))
        ##！
        processing = transforms.Compose(transformations)

        tensor = [processing(img) for img in frame_seq]
        tensor = torch.stack(tensor)
        return tensor

    def get_seq(self, vid_nr, frame_nrs, dkey):
        data_seq = [self.load_data(vid_nr, f_nr, dkey) for f_nr in frame_nrs]
        return self.preprocess_sequence(data_seq, dkey, vid_nr)

    def get_data(self, vid_nr, start):
        if start is None:
            max_start = self.n_images_dict[vid_nr] - self.clip_len + 1
            if max_start == 1:
                start = max_start
            else:
                start = np.random.randint(1, max_start)
        # print('vid_nr:', vid_nr, '\t start:', start)
        frame_nrs = self.get_frame_nrs(vid_nr, start)
        frame_seq = self.get_seq(vid_nr, frame_nrs, 'frame')
        target_size = self.target_size_dict[vid_nr]
        # if self.phase == 'test' and self.source in ('DReye',):
        #     return frame_nrs, frame_seq, target_size
        sal_seq = self.get_seq(vid_nr, frame_nrs, 'sal')

        fix_seq = torch.full(self.target_size, 0, dtype=torch.bool)
        # fix used for nss aucj and aucs
        # fix_seq = self.get_seq(vid_nr, frame_nrs, 'fix')
        # 用 sal_seq替换fix_seq
        return frame_nrs, frame_seq, sal_seq, fix_seq, target_size

    def __getitem__(self, item):
        vid_nr, start = self.samples[item]
        data = self.get_data(vid_nr, start)
        return data

class BDDADataset(Dataset, utils.KwConfigClass):

    img_channels = 1
    n_train_val_videos = 926
    test_vid_nrs = (1127, 1429)  #1110
    frame_rate = 30 
    source = 'BDDA'
    dynamic = True

    def __init__(self,
                 seq_len=12,
                 frame_modulo=5,
                 max_seq_len=1e6,
                 preproc_cfg=None,
                 out_size=(224, 384), phase='train', target_size=(360, 640),
                 debug=False, val_size=200, n_x_val=3, x_val_step=2,
                 x_val_seed=0, seq_per_vid=1, subset=None, verbose=1,
                 n_images_file='BDDA_n_images.dat', seq_per_vid_val=2,
                 sal_offset=None):
        self.phase = phase
        self.train = phase == 'train'
        if not self.train:
            preproc_cfg = {}
        elif preproc_cfg is None:
            preproc_cfg = {}
        preproc_cfg.update({
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        })
        self.preproc_cfg = preproc_cfg
        self.out_size = out_size
        self.debug = debug
        self.val_size = val_size
        self.n_x_val = n_x_val
        self.x_val_step = x_val_step
        self.x_val_seed = x_val_seed
        self.seq_len = seq_len
        self.seq_per_vid = seq_per_vid
        self.seq_per_vid_val = seq_per_vid_val
        self.frame_modulo = frame_modulo
        self.clip_len = seq_len * frame_modulo
        self.subset = subset
        self.verbose = verbose
        self.n_images_file = n_images_file
        self.target_size = target_size
        self.sal_offset = sal_offset
        self.max_seq_len = max_seq_len
        self._dir = None
        self._n_images_dict = None
        self.vid_nr_array = None

        # Evaluation
        if phase in ('eval', 'test'):
            self.seq_len = int(1e6)

        if self.phase in ('test',):
            self.vid_nr_array = list(range(
                self.test_vid_nrs[0], self.test_vid_nrs[1] + 1))
            self.samples, self.target_size_dict = self.prepare_samples()
            return

        # Cross-validation split
        n_videos = self.n_train_val_videos
        assert(self.val_size <= n_videos // self.n_x_val)
        assert(self.x_val_step < self.n_x_val)
        vid_nr_array = np.arange(1, n_videos + 1)
        if self.x_val_seed > 0:
            np.random.seed(self.x_val_seed)
            np.random.shuffle(vid_nr_array)
        val_start = (len(vid_nr_array) - self.val_size) //\
                    (self.n_x_val - 1) * self.x_val_step
        vid_nr_array = vid_nr_array.tolist()
        if not self.train:
            self.vid_nr_array =\
                vid_nr_array[val_start:val_start + self.val_size]
        else:
            del vid_nr_array[val_start:val_start + self.val_size]
            self.vid_nr_array = vid_nr_array

        if self.subset is not None:
            self.vid_nr_array =\
                self.vid_nr_array[:int(len(self.vid_nr_array) * self.subset)]

        self.samples, self.target_size_dict = self.prepare_samples()

    @property
    def n_images_dict(self):
        if self._n_images_dict is None:
            with open(config_path.parent / self.n_images_file, 'r') as f:
                self._n_images_dict = {
                    idx + 1: int(line) for idx, line in enumerate(f)
                    if idx + 1 in self.vid_nr_array}
        return self._n_images_dict

    @property
    def dir(self):
        if self._dir is None:
            self._dir = Path(os.environ["BDDA_DATA_DIR"])
        return self._dir

    @property
    def n_samples(self):
        return len(self.vid_nr_array)

    def __len__(self):
        return len(self.samples)

    def prepare_samples(self):
        samples = []
        too_short = 0
        too_long = 0
        for vid_nr, n_images in self.n_images_dict.items():
            if self.phase in ('eval', 'test'):
                samples += [
                    (vid_nr, offset + 1) for offset in range(self.frame_modulo)]
                continue
            # 帧数过小多大直接跳过
            if n_images < self.clip_len:
                too_short += 1
                continue
            if n_images // self.frame_modulo > self.max_seq_len:
                too_long += 1
                continue
            # 
            if self.phase == 'train':
                samples += [(vid_nr, None)] * self.seq_per_vid
                continue
            elif self.phase == 'valid':
                x = n_images // (self.seq_per_vid_val * 2) - self.clip_len // 2
                start = max(1, x)
                end = min(n_images - self.clip_len, n_images - x)
                samples += [
                    (vid_nr, int(start)) for start in
                    np.linspace(start, end, self.seq_per_vid_val)]
                continue
        # 打印数据集加载的基本信息
        if self.phase not in ('eval', 'test') and self.n_images_dict:
            n_loaded = len(self.n_images_dict) - too_short - too_long
            print(f"{n_loaded} videos loaded "
                  f"({n_loaded / len(self.n_images_dict) * 100:.1f}%)")
            print(f"{too_short} videos are too short "
                  f"({too_short / len(self.n_images_dict) * 100:.1f}%)")
            print(f"{too_long} videos are too long "
                  f"({too_long / len(self.n_images_dict) * 100:.1f}%)")
        target_size_dict = {
            vid_nr: self.target_size for vid_nr in self.n_images_dict.keys()}
        return samples, target_size_dict

    def get_frame_nrs(self, vid_nr, start):
        n_images = self.n_images_dict[vid_nr]
        if self.phase in ('eval', 'test'):
            return list(range(start, n_images + 1, self.frame_modulo))
        return list(range(start, start + self.clip_len, self.frame_modulo))

    def get_data_file(self, vid_nr, f_nr, dkey):
        if dkey == 'frame':
            folder = 'images'
        elif dkey == 'sal':
            folder = 'new_maps'
        elif dkey == 'fix':
            folder = 'fixation'
        else:
            raise ValueError(f'Unknown data key {dkey}')
        ###
        img_path = str(self.dir  / f'{vid_nr:04d}' / folder/ f'{f_nr:04d}.png')
        return img_path

    def load_data(self, vid_nr, f_nr, dkey):
        read_flag = None if dkey == 'frame' else cv2.IMREAD_GRAYSCALE
        data_file = self.get_data_file(vid_nr, f_nr, dkey)
        if read_flag is not None:
            data = cv2.imread(str(data_file), read_flag)
        else:
            data = cv2.imread(str(data_file))
        if data is None:
            raise FileNotFoundError(data_file)
        if dkey == 'frame':
            data = np.ascontiguousarray(data[:, :, ::-1])

        if dkey == 'sal' and self.train and self.sal_offset is not None:
            data += self.sal_offset
            data[0, 0] = 0

        return data

    def preprocess_sequence(self, frame_seq, dkey, vid_nr):
        transformations = []
        if dkey == 'frame':
            transformations.append(transforms.ToPILImage())
            transformations.append(transforms.Resize(
                self.out_size, interpolation=PIL.Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if dkey == 'frame' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif dkey == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        # elif dkey == 'fix':
        #     transformations.append(
        #         transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))
        ##！
        processing = transforms.Compose(transformations)

        tensor = [processing(img) for img in frame_seq]
        tensor = torch.stack(tensor)
        return tensor

    def get_seq(self, vid_nr, frame_nrs, dkey):
        data_seq = [self.load_data(vid_nr, f_nr, dkey) for f_nr in frame_nrs]
        return self.preprocess_sequence(data_seq, dkey, vid_nr)

    def get_data(self, vid_nr, start):
        if start is None:
            max_start = self.n_images_dict[vid_nr] - self.clip_len + 1
            if max_start == 1:
                start = max_start
            else:
                start = np.random.randint(1, max_start)
        frame_nrs = self.get_frame_nrs(vid_nr, start)
        frame_seq = self.get_seq(vid_nr, frame_nrs, 'frame')
        target_size = self.target_size_dict[vid_nr]
        # if self.phase == 'test' and self.source in ('DReye',):
        #     return frame_nrs, frame_seq, target_size
        sal_seq = self.get_seq(vid_nr, frame_nrs, 'sal')

        fix_seq = torch.full(self.target_size, 0, dtype=torch.bool)
        # fix used for nss aucj and aucs
        # fix_seq = self.get_seq(vid_nr, frame_nrs, 'fix')
        # 用 sal_seq替换fix_seq
        return frame_nrs, frame_seq, sal_seq, fix_seq, target_size

    def __getitem__(self, item):
        vid_nr, start = self.samples[item]
        data = self.get_data(vid_nr, start)
        return data