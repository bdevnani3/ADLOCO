import argparse
import os
import pprint
from tokenize import Floatnumber
import warnings
from datetime import datetime

import yaml

from data_loader import dataloaders as dataloader
from train_sklearn import model
from utils import *
import random

##change your data root here
data_root = {"ImageNet": "./datasets/ImageNet/", "Places": "./datasets/Places/"}

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default=None, type=str)
parser.add_argument("--test", default=False, action="store_true")
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--test_open", default=False, action="store_true")
parser.add_argument("--output_logits", default=False)
parser.add_argument("--model_dir", type=str, default=None)
parser.add_argument("--save_feat", type=str, default="")

# Learnable tau
parser.add_argument("--val_as_train", default=False, action="store_true")
parser.add_argument("--seed", default=0, type=int)

#LogReg model config
parser.add_argument("--c", default=1.0, type=float)
parser.add_argument("--max_iter", default=200, type=int)
parser.add_argument("--fit_intercept", default=True, type=bool)
parser.add_argument("--tol", default=1e-5, type=float)
parser.add_argument("--penalty", default="l2", type=str)
parser.add_argument("--class_weight_balanced", default=False, action="store_true")

parser.add_argument("--override_cached_embs", default=False, action="store_true")

parser.add_argument("--expt_str", default="", type=str)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


def update(config, args):
    # Change parameters
    config["training_opt"]["batch_size"] = get_value(
        config["training_opt"]["batch_size"], args.batch_size
    )

    config["training_opt"]["c_regularization"] = args.c
    config["training_opt"]["max_iter"] = args.max_iter
    config["training_opt"]["fit_intercept"] = args.fit_intercept
    config["training_opt"]["tol"] = args.tol
    config["training_opt"]["penalty"] = args.penalty
    config["training_opt"]["class_weight_balanced"] = args.class_weight_balanced

    config["override_cached_embs"] = args.override_cached_embs

    config["model_dir"] = args.model_dir
    config["expt_str"] = args.expt_str
    config["seed"] = args.seed

    return config


# ============================================================================
# LOAD CONFIGURATIONS


def get_log_and_tf_dir(cfg):

    dateTimeObj = datetime.now()
    datetimestr = dateTimeObj.strftime("%d-%b-%Y-%H-%M-%S")
    cfg = cfg.split(".yaml")[0]
    cfg = cfg.split("configs_sklearn/")[1]
    cfg = "_".join(cfg.split("/"))
    if len(args.expt_str) > 0:
        log_cfg = (
            f"/nethome/bdevnani3/flash1/long_tail_lang/results_sklearn/config_"
            + cfg
            + "/"
            + args.expt_str
            + "/"
            + datetimestr
        )
    else:
        log_cfg = (
            f"/nethome/bdevnani3/flash1/long_tail_lang/results_sklearn/config_"
            + cfg
            + "/"
            + datetimestr
        )

    tf_cfg = f"config_" + cfg + "--" + datetimestr

    return log_cfg, tf_cfg


with open(args.cfg) as f:
    config = yaml.safe_load(f)
config = update(config, args)

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits
training_opt = config["training_opt"]
dataset = training_opt["dataset"]

if "log_dir" not in training_opt:
    training_opt["log_dir"], training_opt["tf_folder"] = get_log_and_tf_dir(args.cfg)
    print("Saving results at: {}".format(training_opt["log_dir"]))

    if not os.path.isdir(training_opt["log_dir"]):
        os.makedirs(training_opt["log_dir"])

    copy_current_codebase_to_path(training_opt["log_dir"] + "/src")

print("Loading dataset from: %s" % data_root[dataset.rstrip("_LT")])
pprint.pprint(config)


def split2phase(split):
    if split == "train" and args.val_as_train:
        return "train_val"
    else:
        return split


if not test_mode:

    # Because of weird ImageNet set up
    splits = ["train", "train_plain", "val"]
    if dataset not in ["ImageNet"]:
        splits.append("test")

    data = {}

    if check_config(config["training_opt"], "prompt_set"):
        prompt_set = config["training_opt"]["prompt_set"]
    else:
        config["training_opt"]["prompt_set"] = "ImageNet"
        prompt_set = "ImageNet"

    for x in splits:
        d = dataloader.load_data(
            data_root=data_root[dataset.rstrip("_LT")],
            dataset=dataset,
            phase=split2phase(x),
            batch_size=training_opt["batch_size"],
            sampler_dic=None,
            num_workers=training_opt["num_workers"],
            type=config["dataset_variant"],
            prompt_set=prompt_set,
        )
        data[x] = d[0]
        if x == "train":
            data[x + "_ltcount"] = d[1]

    # CLIP dataloader
    # data = {
    #     x: dataloader.load_data(
    #         data_root=f"/nethome/bdevnani3/flash1/long_tail_lang/datasets/ImageNet_emb/RN50",
    #         phase=x,
    #         batch_size=training_opt["batch_size"],
    #         num_workers=training_opt["num_workers"],
    #     )
    #     for x in splits
    # }
    many_shot_thr = 100
    low_shot_thr = 20
    data["label_categorization"] = {"few": [], "many": [], "medium": []}
    for i in data["train_ltcount"]:
        if data["train_ltcount"][i] > many_shot_thr:
            data["label_categorization"]["many"].append(i)
        elif data["train_ltcount"][i] < low_shot_thr:
            data["label_categorization"]["few"].append(i)
        else:
            data["label_categorization"]["medium"].append(i)

    print(
        "Label categorization: \n Few: {} \n Medium: {} \n Many: {}".format(
            len(data["label_categorization"]["few"]),
            len(data["label_categorization"]["medium"]),
            len(data["label_categorization"]["many"]),
        )
    )

    training_model = model(config, data, test=False)

    training_model.call_algorithm(phase="train")

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    print(
        "Under testing phase, we load training data simply to calculate \
           training data number for each class."
    )

    splits = ["train", "val", "test"]
    test_split = "test"

    data = {}

    for x in splits:
        d = dataloader.load_data(
            data_root=data_root[dataset.rstrip("_LT")],
            dataset=dataset,
            phase=x,
            batch_size=training_opt["batch_size"],
            sampler_dic=None,
            test_open=test_open,
            num_workers=training_opt["num_workers"],
            shuffle=False,
        )
        data[x] = d[0]
        if x == "train":
            data[x + "_ltcount"] = d[1]

    training_model = model(config, data, test=True)

    training_model.call_algorithm(phase=test_split)

print("ALL COMPLETED.")
