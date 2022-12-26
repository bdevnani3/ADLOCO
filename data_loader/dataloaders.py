import io
import logging
import os

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger("global")
import json
import os.path as osp
import time

import random
import requests
import torch

import classes

global_root = "/nethome/bdevnani3/flash1/long_tail_lang"


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert("RGB")
    except IOError:
        logger.info("Failed in loading {}".format(filepath))
    return img


# Image statistics
RGB_statistics = {
    "default": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "clip": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
}

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key="default", override = []):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=224, scale=(0.5, 1), interpolation=BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        ),
    }
    if len(override) > 0:
        print("Overriding data transforms")
        print(override)
        override = [transforms.Resize(224, interpolation=BICUBIC), transforms.CenterCrop(224)] \
            + override + [transforms.ToTensor(), transforms.Normalize(rgb_mean, rbg_std)]
        data_transforms[split] = transforms.Compose(override)

    return data_transforms[split]


# Dataset
class LT_Dataset(Dataset):
    """
    Returns image, label pairs.
    Also returns path of the image and index of the image in the dataset.
    """

    def __init__(self, root, txt, transform=None):
        self.name = "LT_Dataset"
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index, path


# Dataset
class LT_Dataset_transform_mix(Dataset):
    """
    Returns images transformed with different transforms.
    Use with Test set only.
    """

    def __init__(self, root, txt, transform=None):
        self.name = "LT_Dataset"
        self.img_path = []
        self.labels = []
        self.transforms = [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                           transforms.RandomResizedCrop(size=224, scale=(0.9, 1), interpolation=BICUBIC),
                           transforms.RandomResizedCrop(size=224, scale=(0.9, 1), interpolation=BICUBIC),
                           transforms.RandomResizedCrop(size=224, scale=(0.5, 0.55), interpolation=BICUBIC),
                           transforms.RandomResizedCrop(size=224, scale=(0.5, 0.55), interpolation=BICUBIC),]
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        rgb_mean, rgb_std = RGB_statistics["clip"]["mean"], RGB_statistics["clip"]["std"]

        samples = []
        for transform in self.transforms:
            
            transform = [transforms.Resize(224, interpolation=BICUBIC), transforms.CenterCrop(224)] \
                + [transform] + [transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)]
            transform_fn = transforms.Compose(transform)
            samples.append(transform_fn(sample))

        transform = [transforms.Resize(224, interpolation=BICUBIC), transforms.CenterCrop(224), \
            transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)]
        transform_fn = transforms.Compose(transform)
        samples.append(transform_fn(sample))

        return samples[0], samples[1], samples[2],\
                samples[3], samples[4], samples[5], label, index, path


# Dataset
class LT_Dataset_transform_mix(Dataset):
    """
    Returns images transformed with different transforms.
    Use with Test set only.
    """

    def __init__(self, root, txt, transform=None):
        self.name = "LT_Dataset"
        self.img_path = []
        self.labels = []
        self.transforms = [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                           transforms.RandomGrayscale(p=1),
                           transforms.RandomResizedCrop(size=224, scale=(0.97, 1), interpolation=BICUBIC),
                           transforms.RandomResizedCrop(size=224, scale=(0.6, 0.65), interpolation=BICUBIC),
                           transforms.RandomResizedCrop(size=224, scale=(0.6, 0.65), interpolation=BICUBIC),]
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        rgb_mean, rgb_std = RGB_statistics["clip"]["mean"], RGB_statistics["clip"]["std"]

        samples = []
        for transform in self.transforms:
            
            transform = [transforms.Resize(224, interpolation=BICUBIC), transforms.CenterCrop(224)] \
                + [transform] + [transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)]
            transform_fn = transforms.Compose(transform)
            samples.append(transform_fn(sample))

        transform = [transforms.Resize(224, interpolation=BICUBIC), transforms.CenterCrop(224), \
            transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)]
        transform_fn = transforms.Compose(transform)
        samples.append(transform_fn(sample))

        return samples[0], samples[1], samples[2],\
                samples[3], samples[4], samples[5], label, index, path

# Dataset
class LT_Dataset_boosted(Dataset):
    """
    Returns image, label pairs.
    Also index of the image in the dataset.

    Will return path != 1 if the sample needs to be used as the text embedding of the corresponding label.
    When path == 1, sample needs to be used as is.
    """

    def __init__(self, root, txt, transform=None):
        self.name = "LT_Dataset_boosted"
        self.totensor = transforms.ToTensor()
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            l == int(l)
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

        for i in range(1000):
            if freqs[i] <= 100:
                for j in range(82):
                    self.img_path.append(j)
                    self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        if isinstance(path, int):
            sample = self.totensor(np.zeros((224, 224, 3)))
        else:
            with open(path, "rb") as f:
                sample = Image.open(f).convert("RGB")
                path = 1

            if self.transform is not None:
                sample = self.transform(sample)

        return sample, label, index, path


# Dataset
class LT_Dataset_all_prompts(Dataset):
    def __init__(self, root, txt, transform=None, prompt_set="ImageNet"):
        """
        Returns image, label pairs. prompt index indicates the index of the prompt to be used.
        Will be 82 times the regular dataset (since there are 82 prompts).
        """
        self.name = "LT_Dataset_all_prompts"
        self.totensor = transforms.ToTensor()
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            l == int(l)
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

        print("*************************", classes.GENERIC_PROMPT_COLLECTIONS.keys())
        prompt_set_size = len(classes.GENERIC_PROMPT_COLLECTIONS[prompt_set])

        extended_img_path = []
        extended_labels = []
        prompt_index = []
        for image, label in zip(self.img_path, self.labels):
            for i in range(prompt_set_size):
                extended_img_path.append(image)
                extended_labels.append(label)
                prompt_index.append(i)

        self.img_path = extended_img_path
        self.labels = extended_labels
        self.prompt_index = prompt_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        prompt_index = self.prompt_index[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index, prompt_index


# Dataset
class LT_Dataset_random_prompts(Dataset):
    def __init__(self, root, txt, transform=None, prompt_set="ImageNet"):
        """
        Returns image, label pairs.
        """
        self.name = "LT_Dataset_random_prompts"
        self.totensor = transforms.ToTensor()
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            l == int(l)
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

        print("*************************", classes.GENERIC_PROMPT_COLLECTIONS.keys())
        prompt_set_size = len(classes.GENERIC_PROMPT_COLLECTIONS[prompt_set])

        extended_img_path = []
        extended_labels = []
        prompt_index = []
        for image, label in zip(self.img_path, self.labels):
            extended_img_path.append(image)
            extended_labels.append(label)
            prompt_index.append(random.randint(0, prompt_set_size - 1))

        self.img_path = extended_img_path
        self.labels = extended_labels
        self.prompt_index = prompt_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        prompt_index = self.prompt_index[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index, prompt_index


# Dataset
class LT_Dataset_culp_random_prompts(Dataset):
    def __init__(self, root, txt, transform=None):
        """
        Returns image, label pairs.
        """
        self.name = "LT_Dataset_culp_random_prompts"
        self.totensor = transforms.ToTensor()
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            l == int(l)
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

        prompt_set_size = 50

        extended_img_path = []
        extended_labels = []
        prompt_index = []
        for image, label in zip(self.img_path, self.labels):
            extended_img_path.append(image)
            extended_labels.append(label)
            prompt_index.append(random.randint(0, prompt_set_size - 1))

        self.img_path = extended_img_path
        self.labels = extended_labels
        self.prompt_index = prompt_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        prompt_index = self.prompt_index[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index, prompt_index



# Dataset
class LT_Dataset_random_prompts_image_mix(Dataset):
    def __init__(self, root, txt, transform=None, prompt_set="ImageNet"):
        """
        Returns image, label pairs. prompt index indicates the index of the prompt to be used.
        """
        self.name = "LT_Dataset_random_prompts_image_mix"
        self.totensor = transforms.ToTensor()
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            l == int(l)
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

        print("*************************", classes.GENERIC_PROMPT_COLLECTIONS.keys())
        prompt_set_size = len(classes.GENERIC_PROMPT_COLLECTIONS[prompt_set])

        extended_img_path = []
        extended_labels = []
        prompt_index = []
        for image, label in zip(self.img_path, self.labels):
            extended_img_path.append(image)
            extended_labels.append(label)
            prompt_index.append(random.randint(0, prompt_set_size - 1))

            extended_img_path.append(image)
            extended_labels.append(label)
            prompt_index.append(-1)

        self.img_path = extended_img_path
        self.labels = extended_labels
        self.prompt_index = prompt_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        prompt_index = self.prompt_index[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index, prompt_index


# Dataset
class Balanced_text(Dataset):
    def __init__(self, root, txt, transform=None, prompt_set="ImageNet"):
        """
        Returns empty images and labels, with balanced labels. This is simply to
        test the ability of the model to learn from text embeddings.
        """
        self.name = "Balanced_text"
        self.totensor = transforms.ToTensor()
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            l == int(l)
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

        prompt_set_size = len(classes.GENERIC_PROMPT_COLLECTIONS[prompt_set])

        prompt_index = []
        labels = []
        for label in freqs.keys():
            for i in range(prompt_set_size):
                prompt_index.append(i)
                labels.append(label)
        self.labels = labels
        self.prompt_index = prompt_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        label = self.labels[index]
        prompt_index = self.prompt_index[index]

        return torch.tensor([-1]), label, index, prompt_index

class Balanced_text_all(Dataset):
    def __init__(self, root, txt, transform=None, prompt_set="ImageNet"):
        """
        Returns empty images and labels, with balanced labels. This is simply to
        test the ability of the model to learn from text embeddings.
        """
        self.name = "Balanced_text"
        self.totensor = transforms.ToTensor()
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            l == int(l)
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

        ind = []
        prompt_index = []
        labels = []
        for label in freqs.keys():
            for i in range(132):
                prompt_index.append(i)
                labels.append(label)
                ind.append(-1)

        self.labels = labels
        self.prompt_index = prompt_index
        self.ind = ind

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        label = self.labels[index]
        prompt_index = self.prompt_index[index]
        indicator = self.ind[index]

        return indicator, label, index, prompt_index


class Balanced_culp_text(Dataset):
    def __init__(self, root, txt, transform=None, prompt_set="ImageNet"):
        """
        Returns empty images and labels, with balanced labels. This is simply to
        test the ability of the model to learn from text embeddings.
        """
        self.name = "Balanced_culp_text"
        self.totensor = transforms.ToTensor()
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            l == int(l)
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

        prompt_set_size = 50

        prompt_index = []
        labels = []
        for label in freqs.keys():
            for i in range(prompt_set_size):
                prompt_index.append(i)
                labels.append(label)
        self.labels = labels
        self.prompt_index = prompt_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        label = self.labels[index]
        prompt_index = self.prompt_index[index]

        return torch.tensor([-1]), label, index, prompt_index


# Load datasets
def load_data(
    data_root,
    dataset,
    phase,
    batch_size,
    sampler_dic=None,
    num_workers=4,
    test_open=False,
    shuffle=True,
    type=None,
    prompt_set="ImageNet",
    override_transform = [],
):

    if phase == "train_plain":
        txt_split = "train"
    elif phase == "train_val":
        txt_split = "val"
        phase = "train"
    else:
        txt_split = phase
    txt = "%s/data/%s/%s_%s.txt" % (global_root, dataset, dataset, txt_split)
    # txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print("Loading data from %s" % (txt))

    key = "clip"
    rgb_mean, rgb_std = RGB_statistics[key]["mean"], RGB_statistics[key]["std"]

    if phase not in ["train", "val"]:
        transform = get_data_transform("test", rgb_mean, rgb_std, key, override_transform)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std, key, override_transform)

    print("Use data transformation:", transform)

    print("***********************DATASET:", phase, type)

    set_ = LT_Dataset(data_root, txt, transform)

    if phase == "train" and type == "boosted":
        set_ = LT_Dataset_boosted(data_root, txt, transform)
    elif phase == "train" and type == "all_prompts":
        set_ = LT_Dataset_all_prompts(data_root, txt, transform, prompt_set)
    elif phase == "train" and type == "random_prompts":
        set_ = LT_Dataset_random_prompts(data_root, txt, transform, prompt_set)
    elif phase == "train" and type == "random_prompts_shuffled":
        set_ = LT_Dataset_random_prompts(data_root, txt, transform, prompt_set)
    elif phase == "train" and type == "random_prompts_image_mix":
        set_ = LT_Dataset_random_prompts_image_mix(
            data_root, txt, transform, prompt_set
        )
    elif phase == "train" and type == "Balanced_text":
        set_ = Balanced_text(data_root, txt, transform, prompt_set)
    elif phase == "train" and type == "Balanced_culp_text":
        set_ = Balanced_culp_text(data_root, txt, transform, prompt_set)
    elif phase == "train" and type == "Balanced_text_all":
        set_ = Balanced_text_all(data_root, txt, transform, prompt_set)
    elif phase == "train" and type == "culp_random_prompts":
        set_ = LT_Dataset_culp_random_prompts(data_root, txt, transform)
    elif phase == "train" and type == "random_prompts_balanced":
        set_ = LT_Dataset_random_prompts(data_root, txt, transform, prompt_set)
    elif type == "LT_Dataset_transform_mix":
        set_ = LT_Dataset_transform_mix(data_root, txt, transform)

    lt_count = set_.lt_count

    print(phase, len(set_))
    if phase == "test" and test_open:
        open_txt = "%s/data/%s/%s_open.txt" % (global_root, dataset, dataset)
        print("Testing with opensets from %s" % (open_txt))
        open_set_ = LT_Dataset(
            "./data/%s/%s_open" % (dataset, dataset), open_txt, transform
        )
        set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                           num_workers=num_workers)
    else:
        print("No sampler.")
        print("Shuffle is %s." % (shuffle))
        return (
            DataLoader(
                dataset=set_,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            ),
            lt_count,
        )
