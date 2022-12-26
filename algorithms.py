from configparser import NoOptionError
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

from classes import CLASSES, CUSTOM_TEMPLATES, GENERIC_PROMPT_COLLECTIONS
from utils import *


def regular_clip(self, inputs, labels, phase="train"):

    classnames = CLASSES
    templates = self.training_opt["prompt"]

    texts = torch.cat([clip.tokenize(templates.format(c)) for c in classnames])
    texts = texts.cuda()
    zeroshot_weights = self.text_model(texts).float()
    zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(dim=-1, keepdim=True)

    y = self.visual_model(inputs.half()).double()
    x = y
    x = x / x.norm(dim=-1, keepdim=True)

    self.logits = 100.0 * x.double() @ zeroshot_weights.double().t()
    return self.logits


def regular_clip_plus_FT_clip(self, inputs, labels, phase="train"):

    classnames = CLASSES
    templates = self.training_opt["prompt"]

    texts = torch.cat([clip.tokenize(templates.format(c)) for c in classnames])
    texts = texts.cuda()
    zeroshot_weights = self.text_model(texts).float()
    zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(dim=-1, keepdim=True)

    y = self.visual_model(inputs.half()).double()
    x = y
    x = x / x.norm(dim=-1, keepdim=True)

    self.logits = 100.0 * x @ zeroshot_weights.t()
    return self.logits


def clip_with_linear_layer(self, inputs, labels, phase="train"):

    classnames = CLASSES
    templates = CUSTOM_TEMPLATES["ImageNet"]

    texts = torch.cat([clip.tokenize(templates.format(c)) for c in classnames])
    texts = texts.cuda()
    zeroshot_weights = self.text_model(texts).float()
    zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(dim=-1, keepdim=True)

    y = self.visual_model(inputs.half()).double()
    x = y
    x = x / x.norm(dim=-1, keepdim=True)

    logits = 100.0 * x @ zeroshot_weights.t()
    self.logits = self.fusion(logits)
    return self.logits


def add_embs_then_linear1(self, inputs, labels, phase="train"):

    classnames = CLASSES
    templates = CUSTOM_TEMPLATES["ImageNet"]

    texts = torch.cat([clip.tokenize(templates.format(c)) for c in classnames])
    texts = texts.cuda()
    zeroshot_weights = self.text_model(texts).float()
    zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(dim=-1, keepdim=True)

    x = self.visual_model(inputs.half()).double()
    x = x / x.norm(dim=-1, keepdim=True)

    fused = x.unsqueeze(1) + zeroshot_weights.unsqueeze(
        0
    )  # [100, 1, 1024] + [1, 1000, 1024]

    self.logits = self.fusion(fused).squeeze(-1)
    return self.logits


###########################################################################


def image_linear_probe(self, inputs, labels, phase="train"):
    """
    Linear layer of dimension 1024x1000 on top of image embeddings,
    no text used.
    """

    x = self.visual_model(inputs.half()).double()
    x = x / x.norm(dim=-1, keepdim=True)

    self.logits = self.fusion(x)
    return self.logits


def image_linear_probe_proj(self, inputs, labels, phase="train"):
    """
    Linear layer of dimension 1024x100 on top of image embeddings,
    followed by linear layer of dimension 100x1000.
    """

    x = self.visual_model(inputs.half()).double()
    x = x / x.norm(dim=-1, keepdim=True)

    x = self.im_proj(x)
    if check_config(self.training_opt, "use_relu"):
        x = self.relu(x)
    self.logits = self.fusion(x)
    return self.logits


def image_linear_probe_boosted(
    self, inputs, labels, phase="train", paths=None, proj=None
):
    """
    Linear layer of dimension 1024x1000 on top of image embeddings,
    no text used.  To work in tandem with boosted dataset.
    """

    classnames = np.array(CLASSES)
    templates = np.array(GENERIC_PROMPT_COLLECTIONS["ImageNet"])

    if phase == "train" and paths is not None:

        classnames_for_labels = classnames[labels.cpu()]
        prompts_for_labels = templates[paths.cpu()]

        tokens = []
        for p, c in zip(prompts_for_labels, classnames_for_labels):
            tokens.append(clip.tokenize(p.format(c)))
        texts = torch.cat(tokens)

        zeroshot_weights = self.text_model(texts).float()

        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                dim=-1, keepdim=True
            )

        # # # Main difference # # #
        if proj is not None:
            zeroshot_weights = torch.matmul(zeroshot_weights, proj)

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        # Use text embeddings
        idx_im_is_boosted = np.where(paths != 1)
        x[idx_im_is_boosted] = 0.0

        # Use image embeddings
        idx_im_is_not_boosted = np.where(paths == 1)
        zeroshot_weights[idx_im_is_not_boosted] = 0.0

        self.logits = self.fusion(x + zeroshot_weights)

    else:
        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        self.logits = self.fusion(x)
    return self.logits

def image_plus_text_proj(self, inputs, labels, phase="train", lam=None):

    if phase == "train":

        lam = float(self.training_opt["image_emb_weight"]) if lam == None else lam
        classnames = np.array(CLASSES)
        templates = CUSTOM_TEMPLATES["ImageNet"]

        classnames_for_labels = classnames[labels.cpu()]

        texts = torch.cat(
            [clip.tokenize(templates.format(c)) for c in classnames_for_labels]
        )
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts).float()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )

        
        zeroshot_weights = self.text_proj(zeroshot_weights)

        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )

        x = self.visual_model(inputs.half()).float()
        x = x / x.norm(dim=-1, keepdim=True)

        fused = ((lam) * x.double()) + ((1 - lam) * zeroshot_weights)

        fused = fused / fused.norm(dim=-1, keepdim=True)

        self.logits = self.fusion(fused)
        self.proj_weights = self.fusion.module.weight
        self.image_embs = x
        self.projected_text = zeroshot_weights
    else:
        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)
        self.logits = self.fusion(x)
    return self.logits



def image_plus_text(self, inputs, labels, phase="train", lam=None):
    """
    Add every image embedding to the text embedding corresponding to "a photo of a".
    This is the simplest phrase and thus we refer to it as the default.

    """

    if phase == "train":

        lam = float(self.training_opt["image_emb_weight"]) if lam == None else lam
        classnames = np.array(CLASSES)
        templates = CUSTOM_TEMPLATES["ImageNet"]

        classnames_for_labels = classnames[labels.cpu()]

        texts = torch.cat(
            [clip.tokenize(templates.format(c)) for c in classnames_for_labels]
        )
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts).float()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        only_med_and_few = check_config(self.training_opt, "only_med_and_few")
        if only_med_and_few:
            mask = torch.isin(
                labels,
                torch.tensor(self.data["label_categorization"]["many"]).cuda(),
            )
            indices = torch.argwhere(mask)
            zeroshot_weights[indices] = x[indices]

        fused = ((lam) * x) + ((1 - lam) * zeroshot_weights)

        fused = fused / fused.norm(dim=-1, keepdim=True)

        self.logits = self.fusion(fused)
        self.text_embeddings = zeroshot_weights
        self.proj_weights = self.fusion.module.weight
    else:
        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)
        self.logits = self.fusion(x)
    return self.logits


def image_plus_text_all_prompts(
    self, inputs, labels, phase="train", lam=None, paths=None
):
    """
    Add every image embedding to all the text prompts. To be used in tandem with
    the LT_Dataset_all_prompts/LT_Dataset_random_prompts.

    """

    if phase == "train":
        # Get lambda value
        lam = float(self.training_opt["image_emb_weight"]) if lam == None else lam

        # Get classnames and templates
        templates = np.array(
            GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
        )[paths.cpu()]
        classnames_for_labels = np.array(CLASSES)[labels.cpu()]

        # Get text embeddings
        texts = torch.cat(
            [
                clip.tokenize(t.format(c))
                for c, t in zip(classnames_for_labels, templates)
            ]
        )
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts).float()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        # Wherever the path is -1, we replace the text embedding with the image embedding.
        mask = torch.isin(
            paths.cuda(),
            torch.tensor([-1]).cuda(),
        )
        indices = torch.argwhere(mask)
        zeroshot_weights[indices] = x[indices]

        # If we want to do a combination using only med and few, we replace the text embedding with the image embedding for many.
        only_med_and_few = check_config(self.training_opt, "only_med_and_few")
        many_prob = check_config(self.training_opt, "many_prob")
        if "many_prob" not in self.training_opt:
            many_prob = 1

        if only_med_and_few:
            mask = torch.isin(
                labels,
                torch.tensor(self.data["label_categorization"]["many"]).cuda(),
            )
            indices = torch.argwhere(mask)
            rand = torch.randperm(len(indices))
            indices = indices[rand[: int(len(indices) * (many_prob))]]
            zeroshot_weights[indices] = x[indices]

        # elif only_med_and_few:
        #     mask = torch.isin(
        #         labels,
        #         torch.tensor(self.data["label_categorization"]["many"]).cuda(),
        #     )
        #     indices = torch.argwhere(mask)
        #     zeroshot_weights[indices] = x[indices]

        fused = ((lam) * x) + ((1 - lam) * zeroshot_weights)
        fused = fused / fused.norm(dim=-1, keepdim=True)

        self.logits = self.fusion(fused)
    else:

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        self.logits = self.fusion(x)
    return self.logits


def image_plus_text_with_tranformation_all_prompts(
    self, inputs, labels, phase="train", lam=None, paths=None, proj=None
):
    """
    Add every image embedding to all the text prompts. To be used in tandem with
    the LT_Dataset_all_prompts/LT_Dataset_random_prompts. This version uses a learned projection layer.

    """

    if phase == "train":
        # Get lambda value
        lam = float(self.training_opt["image_emb_weight"]) if lam == None else lam

        # Get classnames and templates
        templates = np.array(
            GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
        )[paths.cpu()]
        classnames_for_labels = np.array(CLASSES)[labels.cpu()]

        # Get text embeddings
        texts = torch.cat(
            [
                clip.tokenize(t.format(c))
                for c, t in zip(classnames_for_labels, templates)
            ]
        )
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts).float()

        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                dim=-1, keepdim=True
            )

        # # # Main difference # # #
        if proj is not None:
            zeroshot_weights = torch.matmul(zeroshot_weights, proj)

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        # Wherever the path is -1, we replace the text embedding with the image embedding.
        mask = torch.isin(
            paths.cuda(),
            torch.tensor([-1]).cuda(),
        )
        indices = torch.argwhere(mask)
        zeroshot_weights[indices] = x[indices]

        # If we want to do a combination using only med and few, we replace the text embedding with the image embedding.
        only_med_and_few = check_config(self.training_opt, "only_med_and_few")
        if only_med_and_few:
            mask = torch.isin(
                labels,
                torch.tensor(self.data["label_categorization"]["many"]).cuda(),
            )
            indices = torch.argwhere(mask)
            zeroshot_weights[indices] = x[indices]

        fused = ((lam) * x) + ((1 - lam) * zeroshot_weights)
        fused = fused / fused.norm(dim=-1, keepdim=True)

        self.logits = self.fusion(fused)
    else:

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        self.logits = self.fusion(x)
    return self.logits

import json
def all_text_with_transformation(
    self, inputs, labels, phase="train", lam=None, paths=None, proj=None
):
    """
    Train on text embeddings with a offline learned projection layer.
    """

    if phase == "train":

        # Get classnames and templates
        cupl = json.load(open("/nethome/bdevnani3/flash1/long_tail_lang/data_loader/clip_cupl.json", "r"))

        # Get text embeddings
        texts = torch.cat(
            [
                clip.tokenize(cupl[str(c.item())][t.item()])
                for c, t in zip(labels.cpu(), paths.cpu())
            ]
        )
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts)

        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                dim=-1, keepdim=True
            )

        # # # Main difference # # #
        if proj is not None:
            zeroshot_weights = torch.matmul(zeroshot_weights, proj)

        fused = zeroshot_weights
        # fused = fused / fused.norm(dim=-1, keepdim=True)

        self.logits = self.bn(self.fusion(fused.double()).float())
    else:

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        self.logits = self.fusion(x)
    return self.logits

def text_with_transformation(
    self, inputs, labels, phase="train", lam=None, paths=None, proj=None
):
    """
    Train on text embeddings with a offline learned projection layer.
    """

    if phase == "train":

        # Get classnames and templates
        templates = np.array(
            GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
        )[paths.cpu()]
        classnames_for_labels = np.array(CLASSES)[labels.cpu()]

        # Get text embeddings
        texts = torch.cat(
            [
                clip.tokenize(t.format(c))
                for c, t in zip(classnames_for_labels, templates)
            ]
        )
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts)

        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                dim=-1, keepdim=True
            )

        # # # Main difference # # #
        if proj is not None:
            zeroshot_weights = torch.matmul(zeroshot_weights, proj)

        fused = zeroshot_weights
        # fused = fused / fused.norm(dim=-1, keepdim=True)

        self.logits = self.bn(self.fusion(fused.double()).float())
    else:

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        self.logits = self.fusion(x)
    return self.logits


def image_plus_text_all_prompts_freq_proportional(
    self, inputs, labels, phase="train", lam=None, paths=None
):
    """
    Add every image embedding to all the text prompts. To be used in tandem with
    the LT_Dataset_all_prompts/LT_Dataset_random_prompts.

    """

    if phase == "train":
        # Get lambda value
        lam = float(self.training_opt["image_emb_weight"]) if lam == None else lam

        # Get classnames and templates
        templates = np.array(
            GENERIC_PROMPT_COLLECTIONS[self.training_opt["prompt_set"]]
        )[paths.cpu()]
        classnames_for_labels = np.array(CLASSES)[labels.cpu()]

        # Get text embeddings
        texts = torch.cat(
            [
                clip.tokenize(t.format(c))
                for c, t in zip(classnames_for_labels, templates)
            ]
        )
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts).float()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )

        self.logits = self.fusion(zeroshot_weights)
    else:

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        self.logits = self.fusion(x)
    return self.logits


def image_concat_text(self, inputs, labels, phase="train"):
    """
    Multiply every image embedding to the text embedding to "a photo of a".
    This is the simplest phrase and thus we refer to it as the default.
    """

    if phase == "train":
        classnames = np.array(CLASSES)
        templates = CUSTOM_TEMPLATES["ImageNet"]

        classnames_for_labels = classnames[labels.cpu()]

        texts = torch.cat(
            [clip.tokenize(templates.format(c)) for c in classnames_for_labels]
        )
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts).float()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        fused = torch.cat([x, zeroshot_weights], dim=-1)

        self.logits = self.fusion(fused)
    else:
        m = nn.Softmax(dim=-1)
        batch_size = inputs.shape[0]
        classnames = np.array(CLASSES)
        templates = CUSTOM_TEMPLATES["ImageNet"]
        texts = torch.cat(
            [clip.tokenize(templates.format(c)) for c in classnames]
        ).cuda()
        zeroshot_weights = self.text_model(texts).float()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )  # 1000 x 1024

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)  # batch_size x 1024

        # import pdb; pdb.set_trace()
        y = x.unsqueeze(-1).repeat(1, 1, 1000)  # batch_size x 1024 x 1000
        z = zeroshot_weights.T.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # batch_size x 1024 x 1000

        y = y.permute(0, 2, 1)  # batch_size x 1000 x 1024
        z = z.permute(0, 2, 1)  # batch_size x 1000 x 1024
        fused = torch.cat([y, z], dim=-1)  # batch_size x 1024 x 1000
        out = m(self.fusion(fused))  # batch_size*1000 x 1000
        # out = out * torch.eye(1000,1000).cuda() # batch_size x 1000 x 1000
        out = out.sum(dim=1)  # batch_size x 1000
        self.logits = out
    return self.logits


def image_concat_text_all_prompts(self, inputs, labels, phase="train", paths=None):
    """
    Multiply every image embedding to the text embedding of all the prompts.
    Used in tandem with the LT_all_prompts dataset.
    """

    if phase == "train":
        classnames = np.array(CLASSES)
        templates = np.array(GENERIC_PROMPT_COLLECTIONS["ImageNet"])[paths.cpu()]
        classnames_for_labels = np.array(CLASSES)[labels.cpu()]

        texts = torch.cat(
            [
                clip.tokenize(t.format(c))
                for c, t in zip(classnames_for_labels, templates)
            ]
        )
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts).float()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)

        fused = torch.cat([x, zeroshot_weights], dim=-1)

        self.logits = self.fusion(fused)
    else:
        m = nn.Softmax(dim=-1)
        batch_size = inputs.shape[0]
        classnames = np.array(CLASSES)
        templates = CUSTOM_TEMPLATES["ImageNet"]
        texts = torch.cat(
            [clip.tokenize(templates.format(c)) for c in classnames]
        ).cuda()
        zeroshot_weights = self.text_model(texts).float()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )  # 1000 x 1024

        x = self.visual_model(inputs.half()).double()
        x = x / x.norm(dim=-1, keepdim=True)  # batch_size x 1024

        # import pdb; pdb.set_trace()
        y = x.unsqueeze(-1).repeat(1, 1, 1000)  # batch_size x 1024 x 1000
        z = zeroshot_weights.T.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # batch_size x 1024 x 1000

        y = y.permute(0, 2, 1)  # batch_size x 1000 x 1024
        z = z.permute(0, 2, 1)  # batch_size x 1000 x 1024
        fused = torch.cat([y, z], dim=-1)  # batch_size x 1024 x 1000
        out = m(self.fusion(fused))  # batch_size*1000 x 1000
        # out = out * torch.eye(1000,1000).cuda() # batch_size x 1000 x 1000
        out = out.sum(dim=1)  # batch_size x 1000
        self.logits = out
    return self.logits
