import argparse
import inspect
import os
import pprint
import sys
import warnings
from datetime import datetime

import json
import yaml

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from classes import CLASSES, GENERIC_PROMPT_COLLECTIONS

from data_loader import dataloaders as dataloader
from train import model
from utils import *

##change your data root here
data_root = {"ImageNet": "../datasets/ImageNet/", "Places": "../datasets/Places/"}

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default=None, type=str)
parser.add_argument("--test", default=False, action="store_true")
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--test_open", default=False, action="store_true")
parser.add_argument("--output_logits", default=False)
parser.add_argument("--model_dir", type=str, default=None)
parser.add_argument("--save_feat", type=str, default="")

# KNN testing parameters
parser.add_argument("--knn", default=False, action="store_true")
parser.add_argument("--feat_type", type=str, default="cl2n")
parser.add_argument("--dist_type", type=str, default="l2")

# Learnable tau
parser.add_argument("--val_as_train", default=False, action="store_true")

args = parser.parse_args()


def update(config, args):
    # Change parameters
    config["training_opt"]["batch_size"] = get_value(
        config["training_opt"]["batch_size"], args.batch_size
    )

    # Testing with KNN
    if args.knn and args.test:
        training_opt = config["training_opt"]
        classifier_param = {
            "feat_dim": training_opt["feature_dim"],
            "num_classes": training_opt["num_classes"],
            "feat_type": args.feat_type,
            "dist_type": args.dist_type,
            "log_dir": training_opt["log_dir"],
        }
        classifier = {
            "def_file": "./models/KNNClassifier.py",
            "params": classifier_param,
            "optim_params": config["networks"]["classifier"]["optim_params"],
        }
        config["networks"]["classifier"] = classifier

    config["model_dir"] = args.model_dir
    if "eval_type" not in config["training_opt"]:
        config["training_opt"]["eval_type"] = "image"

    return config


# ============================================================================
# LOAD CONFIGURATIONS

with open(args.cfg) as f:
    config = yaml.safe_load(f)
config = update(config, args)

test_mode = True
output_logits = args.output_logits
training_opt = config["training_opt"]
dataset = training_opt["dataset"]


training_opt["log_dir"], training_opt["tf_folder"] = (
    "/nethome/bdevnani3/flash1/dump",
    "/nethome/bdevnani3/flash1/dump",
)

print("Loading dataset from: %s" % data_root[dataset.rstrip("_LT")])
pprint.pprint(config)


def split2phase(split):
    if split == "train" and args.val_as_train:
        return "train_val"
    else:
        return split


# ============================================================================

out = {}

for prompt in GENERIC_PROMPT_COLLECTIONS["ImageNet"]:
    training_opt["prompt"] = prompt

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    print(
        "Under testing phase, we load training data simply to calculate \
            training data number for each class."
    )

    splits = ["train", "val", "test"]
    test_split = "test"

    # Because of weird ImageNet set up
    if "ImageNet" == training_opt["dataset"]:
        splits = ["train", "val"]
        test_split = "val"

    if args.knn or True:
        splits.append("train_plain")

    data = {}

    for x in splits:
        d = dataloader.load_data(
            data_root=data_root[dataset.rstrip("_LT")],
            dataset=dataset,
            phase=x,
            batch_size=training_opt["batch_size"],
            sampler_dic=None,
            num_workers=training_opt["num_workers"],
            shuffle=False,
        )
        data[x] = d[0]
        if x == "train":
            data[x + "_ltcount"] = d[1]

    training_model = model(config, data, test=True)

    if args.save_feat in ["train_plain", "val", "test"]:
        saveit = True
        test_split = args.save_feat
    else:
        saveit = False

    rsl = training_model.eval(phase=test_split, save_feat=saveit)

    if output_logits:
        training_model.output_logits()

    out[prompt] = rsl

    with open("prompt_strength_checker_output.json", "w") as f:
        json.dump(out, f)


print("ALL COMPLETED.")
