import copy
import enum

# from msilib.schema import SelfReg
import os
import pdb
import pickle
import random
from tabnanny import check, verbose
import time
import warnings
from re import X, template
from readline import set_pre_input_hook

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import clip
from clip import clip
from matplotlib.pyplot import phase_spectrum
from numpy.core.fromnumeric import cumprod
from pytz import NonExistentTimeError
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithms import *
from optimizers import *
from classes import CLASSES, CUSTOM_TEMPLATES, GENERIC_PROMPT_COLLECTIONS
from classes_synsets import SYNS
from diffgrad import diffgrad
from logger import Logger
from utils import *

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import json

import warnings
warnings.filterwarnings("ignore")

import logging

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler



def load_clip_to_cpu_pretrained(visual_backbone):
    backbone_name = visual_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


def load_clip_to_cpu(visual_backbone, checkpoint=None):
    """
    Remember to provide checkpoint of same backbone as visual_backbone.
    """
    model = load_clip_to_cpu_pretrained(visual_backbone)
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location="cuda")
        state_dict = checkpoint["state_dict"]
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
    return model


class model:
    def __init__(self, config, data, test=False):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.training_opt = self.config["training_opt"]
        # self.log_file = open(os.path.join(self.training_opt["log_dir"], "log"), "w")
        logging.basicConfig(filename=os.path.join(self.training_opt["log_dir"], "log"), level=logging.DEBUG)
        self.prompt_set = self.training_opt["prompt_set"]
        self.raw_data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config["shuffle"] if "shuffle" in config else False

        checkpoint = None
        if check_config(self.config["clip"], "checkpoint"):
            checkpoint = self.config["clip"]["checkpoint"]
            self.log(
                "----------> Loading CLIP model from {} <----------".format(checkpoint)
            )
        self.clip_model = load_clip_to_cpu(
            self.config["clip"]["visual_backbone"], checkpoint
        )

        self.writer = SummaryWriter(log_dir="./runs/" + self.training_opt["tf_folder"])

        self.variation = self.training_opt["variation"]

        if check_config(self.training_opt, "proj_path"):
            self.proj = (
                torch.tensor(np.load(self.training_opt["proj_path"])).float()
            )
            self.log(
                "----------> Loading projection matrix from {} <----------".format(
                    self.training_opt["proj_path"]
                )
            )
        else:
            self.proj = None

        self.init_models()

        self.data = {"train": {}, "val": {}, "test": {}}
        with torch.no_grad():
            if not test:
                self.prepare_data(phase="train")
                self.prepare_data(phase="val")
            self.prepare_data(phase="test")

        self.freqs = json.load(open("embedding_datasets/class_frequencies.json", "r"))
        self.freqs = {int(k): v for k, v in self.freqs.items()}

        if check_config(self.config, "center_embeddings"):
            print("Centering embeddings")

            self.data["train"]["image_embeddings"] = self.data["train"]["image_embeddings"] - torch.mean(self.data["train"]["image_embeddings"].float(), axis=0)
            self.data["val"]["image_embeddings"] = self.data["val"]["image_embeddings"] - torch.mean(self.data["val"]["image_embeddings"].float(), axis=0)
            self.data["test"]["image_embeddings"] = self.data["test"]["image_embeddings"] - torch.mean(self.data["test"]["image_embeddings"].float(), axis=0)

            self.data["train"]["text_embeddings"] = self.data["train"]["text_embeddings"] - torch.mean(self.data["train"]["text_embeddings"].float(), axis=0)

        if check_config(self.config, "class_center_embeddings"):

            for i in np.unique(self.data["train"]["labels"]):
                print("Centering embeddings for class {}".format(i))
                if len(self.data["train"]["image_embeddings"][0]) >1:
                    idx = np.where(self.data["train"]["labels"] == i)[0]
                    self.data["train"]["image_embeddings"][idx] = self.data["train"]["image_embeddings"][idx] - torch.mean(self.data["train"]["image_embeddings"][idx], axis=0)
                    idx = np.where(self.data["val"]["labels"] == i)[0]
                    self.data["val"]["image_embeddings"][idx] = self.data["val"]["image_embeddings"][idx] - torch.mean(self.data["val"]["image_embeddings"][idx], axis=0)
                    idx = np.where(self.data["test"]["labels"] == i)[0]
                    self.data["test"]["image_embeddings"][idx] = self.data["test"]["image_embeddings"][idx] - torch.mean(self.data["test"]["image_embeddings"][idx], axis=0)

                if len(self.data["train"]["text_embeddings"][0]) >1:
                    idx = np.where(self.data["train"]["labels"] == i)[0]
                    self.data["train"]["text_embeddings"][idx] = self.data["train"]["text_embeddings"][idx] - torch.mean(self.data["train"]["text_embeddings"][idx], axis=0)

        ## Sampler Logic
        print("** Using SKLEARN sampler? **")
        if check_config(self.config, "sampler"):
            print("** YES **")
            sampler = self.config["sampler"]
            sampling_strategy=self.config["sampling_strategy"]
            if sampler == "oversample":
                if isinstance(sampling_strategy, int):
                    sampling_strategy = {i: max([sampling_strategy,self.freqs[i]]) for i in range(1000)}
                self.sampler = RandomOverSampler(random_state=0, sampling_strategy=sampling_strategy)
            elif sampler == "undersample":
                if isinstance(sampling_strategy, int):
                    sampling_strategy = {i: min([sampling_strategy,self.freqs[i]]) for i in range(1000)}
                self.sampler = RandomUnderSampler(random_state=0, sampling_strategy=sampling_strategy)
            else:
                raise ValueError("Invalid sampler")
            labels = self.data["train"]["labels"]
            indices = np.expand_dims(np.array(range(len(labels))), axis=1)
            indices_res, labels_res = self.sampler.fit_resample(indices, labels)
            indices_res = indices_res.flatten()
            # import pdb; pdb.set_trace()
            self.data["train"]["image_embeddings"] = self.data["train"]["image_embeddings"][indices_res]
            self.data["train"]["labels"] = labels[indices_res]
            self.data["train"]["text_embeddings"] = self.data["train"]["text_embeddings"][indices_res]
            print(self.data["train"]["image_embeddings"].shape)
        else:
            print("** NO **")

    def init_models(self):

        self.visual_model = torch.nn.DataParallel(self.clip_model.visual).cuda()
        text_model = TextEncoder(self.clip_model)
        self.text_model = torch.nn.DataParallel(text_model).cuda()
        self.C = self.training_opt["c_regularization"]
        self.max_iter = check_config(self.training_opt,"max_iter")
        self.fit_intercept = check_config(self.training_opt,"fit_intercept")
        self.tol = check_config(self.training_opt,"tol")
        self.penalty = check_config(self.training_opt,"penalty")
        balanced = check_config(self.training_opt,"class_weight_balanced")
        # self.lr = LogisticRegression(random_state=0,verbose=1, n_jobs=-1, max_iter=100, C=np.inf, tol=1e-6)
        self.lr = LogisticRegression(random_state=self.config["seed"], verbose=2, n_jobs=-1, max_iter=self.max_iter, \
            C=self.C , tol=self.tol, penalty=self.penalty, fit_intercept=self.fit_intercept, solver=self.config["optimizer_variant"], class_weight="balanced" if balanced else None)
        # self.lr = LogisticRegressionCV(random_state=0,verbose=1, n_jobs=-1, max_iter=500, \
        #     cv=10, solver="lbfgs")

    def prepare_data(self, phase="train"):

        self.log("Preparing Dataset {} for phase {}".format(self.config["dataset_variant"], phase))

        potential_data_path = os.path.join("embedding_datasets/clip", "{}_{}_{}".format(self.config["dataset_variant"], phase, self.prompt_set))
        str_image_embeddings = os.path.join(potential_data_path, "image_embeddings.pt")
        str_text_embeddings = os.path.join(potential_data_path, "text_embeddings.pt")
        str_labels = os.path.join(potential_data_path, "labels.pt")

        if os.path.isdir(potential_data_path) and not self.config["override_cached_embs"]:
            print ("Loading dataset from {}".format(potential_data_path))
            self.data[phase]["image_embeddings"] = torch.load(str_image_embeddings, map_location=torch.device('cpu'))
            self.data[phase]["text_embeddings"] = torch.load(str_text_embeddings, map_location=torch.device('cpu'))
            self.data[phase]["labels"] = torch.load(str_labels, map_location=torch.device('cpu'))
            print(len(self.data[phase]["image_embeddings"]))
            return

        image_embeddings = []
        text_embeddings = []
        labels = []
        texts = []

        if phase == "train":
            if self.config["dataset_variant"] == "LT_Dataset":

                for im, label, _, path in tqdm(self.raw_data[phase]):
                    x = self.visual_model(im.half()).float()
                    x = x / x.norm(dim=-1, keepdim=True)
                    image_embeddings.append(x)
                    labels.append(label)
                    text_embeddings.append(torch.ones_like(x))

            elif self.config["dataset_variant"] == "LT_Dataset_more_transforms":

                for im, label, _, path in tqdm(self.raw_data[phase]):
                    x = self.visual_model(im.half()).float()
                    x = x / x.norm(dim=-1, keepdim=True)
                    image_embeddings.append(x)
                    labels.append(label)
                    text_embeddings.append(torch.ones_like(x))

            elif self.config["dataset_variant"] == "LT_Dataset_no_jitter":

                for im, label, _, path in tqdm(self.raw_data[phase]):
                    x = self.visual_model(im.half()).float()
                    x = x / x.norm(dim=-1, keepdim=True)
                    image_embeddings.append(x)
                    labels.append(label)
                    text_embeddings.append(torch.ones_like(x))

            elif self.config["dataset_variant"] == "Balanced_text":

                for im, label, _, path in tqdm(self.raw_data[phase]):
                    templates = np.array(
                        GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
                    )[path]
                    classnames_for_labels = np.array(CLASSES)[label]
                    texts = torch.cat(
                        [
                            clip.tokenize(t.format(c))
                            for c, t in zip(classnames_for_labels, templates)
                        ]
                    )
                    zeroshot_weights = self.text_model(texts).float()
                    zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                            dim=-1, keepdim=True
                        )
                    image_embeddings.append(im)
                    labels.append(label)
                    text_embeddings.append(zeroshot_weights)

            elif self.config["dataset_variant"] == "Balanced_culp_text":

                all_prompts = json.load(open("data_loader/cupl_image_prompts.json", "r"))

                for im, label, _, path in tqdm(self.raw_data[phase]):
                    # import pdb; pdb.set_trace()
                    classnames_for_labels = np.array(CLASSES)[label]

                    texts = [all_prompts[c][p.item()] for c,p in zip(classnames_for_labels, path)]
                    texts = torch.cat([clip.tokenize(t) for t in texts])
                    zeroshot_weights = self.text_model(texts).float()
                    zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                            dim=-1, keepdim=True
                        )
                    image_embeddings.append(im)
                    labels.append(label)
                    text_embeddings.append(zeroshot_weights)

            elif self.config["dataset_variant"] == "wiki":
                desc_path = "/nethome/bdevnani3/flash1/long_tail_lang/data_loader/imagenet/wiki/desc_{}.txt"

                for label in range(1000):
                    label_desc_path = desc_path.format(label)
                    f = open(label_desc_path)
                    for line in f:
                        line = line.strip()
                        if "==" in line:
                            continue
                        if len(line) == 0:
                            continue
                        line = line[:76]
                        texts = clip.tokenize(line)
                        texts = texts.cuda()
                        zeroshot_weights = self.text_model(texts).float()
                        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                            dim=-1, keepdim=True
                        )
                        image_embeddings.append(torch.tensor([-1.0]))
                        labels.append(label)
                        text_embeddings.append(zeroshot_weights)
                f.close()

            elif self.config["dataset_variant"] == "all_texts":

                for label in tqdm(range(1000)):
                    for p in range(82):
                        templates = np.array(
                            GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
                        )[p]
                        classnames_for_labels = np.array(CLASSES)[label]
                        texts = clip.tokenize(templates.format(classnames_for_labels))
                        zeroshot_weights = self.text_model(texts).float()
                        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                                dim=-1, keepdim=True
                            )
                        image_embeddings.append(torch.tensor([-1.0]))
                        labels.append(torch.tensor([label]))
                        text_embeddings.append(zeroshot_weights)

                print(len(text_embeddings))
                print(len(labels))

                all_prompts = json.load(open("data_loader/cupl_image_prompts.json", "r"))

                for label in tqdm(range(1000)):
                    for p in range(50):
                        # import pdb; pdb.set_trace()
                        classnames_for_labels = np.array(CLASSES)[label]

                        texts = [all_prompts[classnames_for_labels][p]]
                        texts = torch.cat([clip.tokenize(t) for t in texts])
                        zeroshot_weights = self.text_model(texts).float()
                        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                                dim=-1, keepdim=True
                            )
                        image_embeddings.append(torch.tensor([-1.0]))
                        labels.append(torch.tensor([label]))
                        text_embeddings.append(zeroshot_weights)

                print(len(text_embeddings))
                print(len(labels))

                desc_path = "/nethome/bdevnani3/flash1/long_tail_lang/data_loader/imagenet/wiki/desc_{}.txt"

                for label in tqdm(range(1000)):
                    label_desc_path = desc_path.format(label)
                    f = open(label_desc_path)
                    for line in f:
                        line = line.strip()
                        if "==" in line:
                            continue
                        if len(line) == 0:
                            continue
                        line = line[:76]
                        texts = clip.tokenize(line)
                        texts = texts.cuda()
                        zeroshot_weights = self.text_model(texts).float()
                        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                            dim=-1, keepdim=True
                        )
                        image_embeddings.append(torch.tensor([-1.0]))
                        labels.append(torch.tensor([label]))
                        text_embeddings.append(zeroshot_weights)
                f.close()

                print(len(text_embeddings))
                print(len(labels))

            elif self.config["dataset_variant"] == "clip_and_cupl":

                for label in tqdm(range(1000)):
                    for p in range(82):
                        templates = np.array(
                            GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
                        )[p]
                        classnames_for_labels = np.array(CLASSES)[label]
                        texts = clip.tokenize(templates.format(classnames_for_labels))
                        zeroshot_weights = self.text_model(texts).float()
                        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                                dim=-1, keepdim=True
                            )
                        image_embeddings.append(torch.tensor([-1.0]))
                        labels.append(torch.tensor([label]))
                        text_embeddings.append(zeroshot_weights)

                print(len(text_embeddings))
                print(len(labels))

                all_prompts = json.load(open("data_loader/cupl_image_prompts.json", "r"))

                for label in tqdm(range(1000)):
                    for p in range(50):
                        # import pdb; pdb.set_trace()
                        classnames_for_labels = np.array(CLASSES)[label]

                        texts = [all_prompts[classnames_for_labels][p]]
                        texts = torch.cat([clip.tokenize(t) for t in texts])
                        zeroshot_weights = self.text_model(texts).float()
                        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                                dim=-1, keepdim=True
                            )
                        image_embeddings.append(torch.tensor([-1.0]))
                        labels.append(torch.tensor([label]))
                        text_embeddings.append(zeroshot_weights)

                print(len(text_embeddings))
                print(len(labels))

            elif self.config["dataset_variant"] == "clip_and_cupl":

                for label in tqdm(range(1000)):
                    for p in range(82):
                        templates = np.array(
                            GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
                        )[p]
                        classnames_for_labels = random.sample(SYNS[label])
                        texts = clip.tokenize(templates.format(classnames_for_labels))
                        zeroshot_weights = self.text_model(texts).float()
                        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                                dim=-1, keepdim=True
                            )
                        image_embeddings.append(torch.tensor([-1.0]))
                        labels.append(torch.tensor([label]))
                        text_embeddings.append(zeroshot_weights)

                print(len(text_embeddings))
                print(len(labels))

            elif self.config["dataset_variant"] == "clip_synonyms":

                for label in tqdm(range(1000)):
                    for p in range(82):
                        templates = np.array(
                            GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
                        )[p]
                        classnames_for_labels = np.array(CLASSES)[label]
                        texts = clip.tokenize(templates.format(classnames_for_labels))
                        zeroshot_weights = self.text_model(texts).float()
                        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                                dim=-1, keepdim=True
                            )
                        image_embeddings.append(torch.tensor([-1.0]))
                        labels.append(torch.tensor([label]))
                        text_embeddings.append(zeroshot_weights)

                print(len(text_embeddings))
                print(len(labels))

            elif self.config["dataset_variant"] == "random_prompts":

                for im, label, _, path in tqdm(self.raw_data[phase]):

                    x = self.visual_model(im.half()).float()
                    x = x / x.norm(dim=-1, keepdim=True)

                    templates = np.array(
                        GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
                    )[path]
                    classnames_for_labels = np.array(CLASSES)[label]
                    texts = torch.cat(
                        [
                            clip.tokenize(t.format(c))
                            for c, t in zip(classnames_for_labels, templates)
                        ]
                    )
                    zeroshot_weights = self.text_model(texts).float()
                    zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                            dim=-1, keepdim=True
                        )


                    image_embeddings.append(x)
                    labels.append(label)
                    text_embeddings.append(zeroshot_weights)

            elif self.config["dataset_variant"] == "random_prompts_balanced":

                im_embeds_by_label = {i:[] for i in range(1000)}
                text_embeds_by_label = {i:[] for i in range(1000)}

                # Collect all the image embeddings and sort by label
                for im, label, _, path in tqdm(self.raw_data[phase]):
                    x = self.visual_model(im.half()).float()
                    x = x / x.norm(dim=-1, keepdim=True)

                    for i in range(len(label)):
                        im_embeds_by_label[label[i].item()].append(x[i])

                templates = np.array(
                    GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
                )
                for label in tqdm(range(1000)):

                    classname_for_label = np.array(CLASSES)[label]
                    texts = torch.cat(
                        [
                            clip.tokenize(t.format(classname_for_label))
                            for t in templates
                        ]
                    )
                    zeroshot_weights = self.text_model(texts).float()
                    zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                            dim=-1, keepdim=True
                        )

                    text_embeds_by_label[label] = zeroshot_weights

                # Generate the balanced dataset

                for label in tqdm(range(1000)):
                    for _ in range(1200):

                        idx_im = np.random.randint(len(im_embeds_by_label[label]))
                        idx_text = np.random.randint(len(text_embeds_by_label[label]))

                        image_embeddings.append(im_embeds_by_label[label][idx_im].unsqueeze(0))
                        labels.append(torch.tensor([label]).unsqueeze(0))
                        text_embeddings.append(text_embeds_by_label[label][idx_text].unsqueeze(0))

                # import pdb; pdb.set_trace()

            elif self.config["dataset_variant"] == "random_prompts_shuffled":

                for im, label, _, path in tqdm(self.raw_data[phase]):
                    # import pdb; pdb.set_trace()

                    x = self.visual_model(im.half()).float()
                    x = x / x.norm(dim=-1, keepdim=True)

                    if check_config(self.config, "apply_offset"):
                        offset = (label + 10) % 1000
                    else:
                        offset = label

                    template = "This is a photo of a {}"
                    classnames_for_labels = np.array(CLASSES)[offset]
                    texts = torch.cat(
                        [
                            clip.tokenize(template.format(c))
                            for c in classnames_for_labels
                        ]
                    )
                    zeroshot_weights = self.text_model(texts).float()
                    zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                            dim=-1, keepdim=True
                        )


                    image_embeddings.append(x)
                    labels.append(label)
                    text_embeddings.append(zeroshot_weights)

            elif self.config["dataset_variant"] == "culp_random_prompts":

                all_prompts = json.load(open("data_loader/cupl_image_prompts.json", "r"))

                for im, label, _, path in tqdm(self.raw_data[phase]):

                    x = self.visual_model(im.half()).float()
                    x = x / x.norm(dim=-1, keepdim=True)

                    classnames_for_labels = np.array(CLASSES)[label]

                    texts = [random.choice(all_prompts[c]) for i,c in enumerate(classnames_for_labels)]
                    texts = torch.cat(
                        [
                            clip.tokenize(texts)
                        ]
                    )
                    zeroshot_weights = self.text_model(texts).float()
                    zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                            dim=-1, keepdim=True
                        )


                    image_embeddings.append(x)
                    labels.append(label)
                    text_embeddings.append(zeroshot_weights)

            elif self.config["dataset_variant"] == "image_only_and_text_only":

                for im, label, _, path in tqdm(self.raw_data[phase]):

                    x = self.visual_model(im.half()).float()
                    x = x / x.norm(dim=-1, keepdim=True)

                    templates = np.array(
                        GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
                    )[path]
                    classnames_for_labels = np.array(CLASSES)[label]
                    texts = torch.cat(
                        [
                            clip.tokenize(t.format(c))
                            for c, t in zip(classnames_for_labels, templates)
                        ]
                    )
                    zeroshot_weights = self.text_model(texts).float()
                    zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                            dim=-1, keepdim=True
                        )


                    image_embeddings.append(x)
                    labels.append(label)
                    text_embeddings.append(zeroshot_weights)

        else:
            for im, label, _, path in tqdm(self.raw_data[phase]):
                x = self.visual_model(im.half()).float()
                x = x / x.norm(dim=-1, keepdim=True)
                image_embeddings.append(x)
                labels.append(label)
                text_embeddings.append(torch.tensor([-1.0]))

        # import pdb; pdb.set_trace()
        self.data[phase]["image_embeddings"] = torch.cat(image_embeddings, dim=0).cpu()
        self.data[phase]["text_embeddings"] = torch.cat(text_embeddings, dim=0).cpu()
        self.data[phase]["labels"] = torch.cat(labels, dim=0).cpu()
        os.makedirs(potential_data_path, exist_ok=True)
        torch.save(self.data[phase]["image_embeddings"],str_image_embeddings)
        torch.save(self.data[phase]["text_embeddings"],str_text_embeddings)
        torch.save(self.data[phase]["labels"],str_labels)
        print(len(self.data[phase]["image_embeddings"]))

    def call_algorithm(self, phase="train"):

        self.log(json.dumps(self.config))

        start_time = time.time()

        variation = self.training_opt["variation"]

        if phase == "test":
            self.clf = self.load_model()

        # Algorithms
        self.log(f"==== Training ====")
        if variation == "image_linear_probe":
            self.image_linear_probe(phase=phase)
        elif variation == "text_only":
            self.text_only(phase=phase)
        elif variation == "image_plus_text":
            lam = check_config(self.training_opt, "lam")
            self.conv_comb(phase=phase, lam=lam)
        elif variation == "selective_image_plus_text":
            lam = check_config(self.training_opt, "lam")
            self.selective_conv_comb(phase=phase, lam=lam)
        elif variation == "image_plus_text_weighted":
            self.conv_comb_weighted(phase=phase)
        elif variation == "image_plus_text_only_med_and_few":
            lam = check_config(self.training_opt, "lam")
            self.image_plus_text_only_med_and_few(phase=phase, lam=lam)
        elif variation == "text_proj":
            lam = check_config(self.training_opt, "lam")
            self.text_proj(phase=phase, lam=lam)
        elif variation == "conv_comb_randomize_lam":
            self.conv_comb_randomize_lam(phase=phase)
        elif variation == "conv_comb_randomize_lam_3x":
            self.conv_comb_randomize_lam_3x(phase=phase)
        elif variation == "conv_comb_3x":
            lam = check_config(self.training_opt, "lam")
            self.conv_comb_3x(phase=phase, lam=lam)

        if phase == "train":
            self.log("Saving results at {}".format(self.training_opt["log_dir"]))
            self.model_save(self.clf, filename="final_model.sav")
            self.evaluate(phase="train")
            self.evaluate(phase="val")
        test_acc = self.evaluate(phase="test")

        end_time = time.time()
        total_time = end_time - start_time
        self.log("Time taken: {}".format(total_time))

        return test_acc



    def evaluate(self, phase="train"):
        
        self.log(f"==== {phase} Evaluation ====")
        predictions = self.clf.predict(self.data[phase]["image_embeddings"])
        labels = self.data[phase]["labels"].numpy()
        avg_acc = mic_acc_cal(predictions, labels)
        shotwise_acc = shot_acc(predictions, labels, self.freqs)
        self.log("{} | {} | {} | {}".format(shotwise_acc["many_shot"]*100, shotwise_acc["median_shot"]*100, shotwise_acc["low_shot"]*100, avg_acc*100))
        return avg_acc
            

    def model_save(self, model, filename="final_model.sav"):
        path = os.path.join(self.training_opt["log_dir"], filename)
        pickle.dump(model, open(path, 'wb'))

    def load_model(self):
        model_dir = self.training_opt["log_dir"]
        path = os.path.join(self.training_opt["log_dir"], "final_model.sav")
        if os.path.isfile(model_dir + path):
            self.loaded_model = pickle.load(open(path, 'rb'))
            return self.loaded_model
        else:
            self.log("No model found at {}".format(path))

    def log(self, line):
        print(line)
        logging.info(line + "\n")

########### Algorithms ############

    def image_linear_probe(self, phase="train"):

        if phase == "train" or phase == "val":
            self.clf = self.lr.fit(self.data["train"]["image_embeddings"], self.data["train"]["labels"])
        else:
            pass

    def text_only(self, phase="train"):

        if phase == "train" or phase == "val":
            
            if self.proj is not None:
                projected_texts = np.matmul(self.data["train"]["text_embeddings"], self.proj)
                projected_texts = projected_texts / np.linalg.norm(projected_texts, axis=-1, keepdims=True)

            else:
                projected_texts = self.data["train"]["text_embeddings"]

            self.clf = self.lr.fit(projected_texts, self.data["train"]["labels"])

            # for evaluation
            self.data["train"]["image_embeddings"] = projected_texts.cpu()
        else:
            pass

    def conv_comb(self, phase="train", lam=0.75):

        if phase == "train" or phase == "val":
            training_data = self.data["train"]["image_embeddings"] * lam + self.data["train"]["text_embeddings"] * (1 - lam)
            self.clf = self.lr.fit(training_data, self.data["train"]["labels"])
            # for evaluation
            self.data["train"]["image_embeddings"] = training_data.cpu()

    def selective_conv_comb(self, phase="train"):

        if phase == "train" or phase == "val":
            lam = np.zeros(self.data["train"]["image_embeddings"].shape[0])
            
            training_data = self.data["train"]["image_embeddings"] * lam + self.data["train"]["text_embeddings"] * (1 - lam)
            self.clf = self.lr.fit(training_data, self.data["train"]["labels"])
            # for evaluation
            self.data["train"]["image_embeddings"] = training_data.cpu()

    def conv_comb_weighted(self, phase="train"):

        if phase == "train" or phase == "val":
            # Reweighting
            lam = np.ones(self.data["train"]["image_embeddings"].shape[0])
            freqs = np.array([self.freqs[i.item()] for i in self.data["train"]["labels"]])
            lam[freqs < 20] = 0.8
            lam[np.logical_and(freqs > 20,  freqs < 50)] = 0.8
            lam[np.logical_and(freqs > 50,  freqs < 200)] = 0.8
            lam[np.logical_and(freqs > 200,  freqs < 500)] = 0.9
            lam[np.logical_and(freqs > 500,  freqs < 1000)] = 1
            lam[freqs > 1000] = 1
            lam = lam[:,None]

            training_data = self.data["train"]["image_embeddings"] * lam + self.data["train"]["text_embeddings"] * (1 - lam)
            self.clf = self.lr.fit(training_data, self.data["train"]["labels"])
            # for evaluation
            self.data["train"]["image_embeddings"] = training_data.cpu()
        else:
            pass

    def text_proj_and_conv_comb_weighted(self, phase="train"):

        if phase == "train" or phase == "val":
            # Reweighting
            lam = np.ones(self.data["train"]["image_embeddings"].shape[0])
            freqs = np.array([self.freqs[i.item()] for i in self.data["train"]["labels"]])
            lam[freqs < 20] = 0.75
            lam[np.logical_and(freqs > 20,  freqs < 50)] = 0.78
            lam[np.logical_and(freqs > 50,  freqs < 200)] = 0.8
            lam[np.logical_and(freqs > 200,  freqs < 500)] = 0.85
            lam[np.logical_and(freqs > 500,  freqs < 1000)] = 1
            lam[freqs > 1000] = 1
            lam = lam[:,None]

            if self.proj is not None:
                text_embeddings = self.data["train"]["text_embeddings"]
                projected_texts = np.matmul(text_embeddings, self.proj)
                projected_texts = projected_texts / np.linalg.norm(projected_texts, axis=-1, keepdims=True)
            else:
                projected_texts = self.data["train"]["text_embeddings"]

            training_data = self.data["train"]["image_embeddings"] * lam + projected_texts * (1 - lam)
            self.clf = self.lr.fit(training_data, self.data["train"]["labels"])
            # for evaluation
            self.data["train"]["image_embeddings"] = training_data.cpu()
        else:
            pass


    def text_proj_and_conv_comb_weighted(self, phase="train"):

        if phase == "train" or phase == "val":
            # Reweighting
            lam = np.ones(self.data["train"]["image_embeddings"].shape[0])
            freqs = np.array([self.freqs[i.item()] for i in self.data["train"]["labels"]])
            lam[freqs < 20] = 0.75
            lam[np.logical_and(freqs > 20,  freqs < 50)] = 0.78
            lam[np.logical_and(freqs > 50,  freqs < 200)] = 0.8
            lam[np.logical_and(freqs > 200,  freqs < 500)] = 0.85
            lam[np.logical_and(freqs > 500,  freqs < 1000)] = 1
            lam[freqs > 1000] = 1
            lam = lam[:,None]

            if self.proj is not None:
                text_embeddings = self.data["train"]["text_embeddings"]
                projected_texts = np.matmul(text_embeddings, self.proj)
                projected_texts = projected_texts / np.linalg.norm(projected_texts, axis=-1, keepdims=True)
            else:
                projected_texts = self.data["train"]["text_embeddings"]

            training_data = self.data["train"]["image_embeddings"] * lam + projected_texts * (1 - lam)
            self.clf = self.lr.fit(training_data, self.data["train"]["labels"])
            # for evaluation
            self.data["train"]["image_embeddings"] = training_data.cpu()
        else:
            pass


    def image_plus_text_only_med_and_few(self, phase="train", lam=0.75):

        if phase == "train" or phase == "val":
            training_data = self.data["train"]["image_embeddings"] * lam + self.data["train"]["text_embeddings"] * (1 - lam)

            many_inds = []
            for i,label in enumerate(self.data["train"]["labels"]):
                if self.freqs[label.item()] > 100:
                    many_inds.append(i)

            training_data[np.array(many_inds)] = self.data["train"]["image_embeddings"][np.array(many_inds)]

            self.clf = self.lr.fit(training_data, self.data["train"]["labels"])
            # for evaluation
            self.data["train"]["image_embeddings"] = training_data.cpu()

    
    def text_proj(self, phase="train", lam=0.75):

        if phase == "train" or phase == "val":

            text_embeddings = self.data["train"]["text_embeddings"]
            projected_texts = np.matmul(text_embeddings, self.proj)
            projected_texts = projected_texts / np.linalg.norm(projected_texts, axis=-1, keepdims=True)
            training_data = self.data["train"]["image_embeddings"] * lam + projected_texts * (1 - lam)

            self.clf = self.lr.fit(training_data, self.data["train"]["labels"])
            # for evaluation
            self.data["train"]["image_embeddings"] = training_data.cpu()


    def add_bias(self, phase="train", lam=0.75):

        if phase == "train" or phase == "val":
            bias = self.data["train"]["text_embeddings"]
            training_data = self.data["train"]["image_embeddings"] * lam + bias * (1 - lam)
            self.clf = self.lr.fit(training_data, self.data["train"]["labels"])
            # for evaluation
            self.data["train"]["image_embeddings"] = training_data.cpu()

    def conv_comb_randomize_lam(self, phase="train"):

        if phase == "train" or phase == "val":
            lam = np.array([random.uniform(0, 1) for _ in range(len(self.data["train"]["image_embeddings"]))])
            training_data = self.data["train"]["image_embeddings"] * lam[:, np.newaxis] + self.data["train"]["text_embeddings"] * ((1 - lam)[:, np.newaxis])
            self.clf = self.lr.fit(training_data, self.data["train"]["labels"])
            # for evaluation
            self.data["train"]["image_embeddings"] = training_data.cpu()

    def conv_comb_randomize_lam_3x(self, phase="train"):
        """Randomly choose a lambda for each of the image embeddings + text combinations.
        Uses only image, only text and a combination of the two for every instance."""

        if phase == "train" or phase == "val":
            training_set_size = len(self.data["train"]["image_embeddings"])
            lam = np.array([random.uniform(0, 1) for _ in range(training_set_size)])
            training_data_1 = self.data["train"]["image_embeddings"] * lam[:, np.newaxis] + self.data["train"]["text_embeddings"] * ((1 - lam)[:, np.newaxis])
            training_data_2 = self.data["train"]["image_embeddings"]
            training_data_3 = self.data["train"]["text_embeddings"]
            labels = np.concatenate((self.data["train"]["labels"], self.data["train"]["labels"], self.data["train"]["labels"]), axis=0)
            training_data = np.concatenate((training_data_1, training_data_2, training_data_3), axis=0)
            self.clf = self.lr.fit(training_data, labels)
            # for evaluation
            self.data["train"]["image_embeddings"] = training_data
            self.data["train"]["labels"] = torch.tensor(labels)

    def conv_comb_3x(self, phase="train", lam=0.75):

        if phase == "train" or phase == "val":
            training_data_1 = self.data["train"]["image_embeddings"] * lam + self.data["train"]["text_embeddings"] * ((1 - lam))
            training_data_2 = self.data["train"]["image_embeddings"]
            training_data_3 = self.data["train"]["text_embeddings"]
            labels = np.concatenate((self.data["train"]["labels"], self.data["train"]["labels"], self.data["train"]["labels"]), axis=0)
            training_data = np.concatenate((training_data_1, training_data_2, training_data_3), axis=0)
            self.clf = self.lr.fit(training_data, labels)
            # for evaluation
            self.data["train"]["image_embeddings"] = training_data
            self.data["train"]["labels"] = torch.tensor(labels)

        

