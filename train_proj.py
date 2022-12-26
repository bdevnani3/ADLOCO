import copy
import enum

# from msilib.schema import SelfReg
import os
import pdb
import pickle
import random
from tabnanny import verbose
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
from diffgrad import diffgrad
from logger import Logger
from utils import *

from sklearn.linear_model import LogisticRegression


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
        self.model_opt = self.config["model"]
        self.data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config["shuffle"] if "shuffle" in config else False

        checkpoint = None
        if check_config(self.model_opt["clip"]["params"], "checkpoint"):
            checkpoint = self.model_opt["clip"]["params"]["checkpoint"]
            print(
                "----------> Loading CLIP model from {} <----------".format(checkpoint)
            )
        self.clip_model = load_clip_to_cpu(
            self.model_opt["clip"]["params"]["visual_backbone"], checkpoint
        )

        self.writer = SummaryWriter(log_dir="./runs/" + self.training_opt["tf_folder"])

        # Setup logger
        self.logger = Logger(self.training_opt["log_dir"])

        self.optimizer_variant = (
            config["optimizer_variant"] if "optimizer_variant" in config else None
        )

        # Initialize model
        self.init_models()

        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            print("Using steps for training.")
            self.training_data_num = len(self.data["train"].dataset)
            self.epoch_steps = int(
                self.training_data_num / self.training_opt["batch_size"]
            )

            # Initialize model optimizer and scheduler
            print("Initializing model optimizer.")
            self.scheduler_params = self.training_opt["scheduler_params"]
            self.model_optimizer, self.model_optimizer_scheduler = init_optimizers(
                self, self.model_optim_params_list
            )
            init_criterions(self)

            # Set up log file
            self.log_file = os.path.join(self.training_opt["log_dir"], "log.txt")
            if os.path.isfile(self.log_file):
                os.remove(self.log_file)
            self.logger.log_cfg(self.config)
        else:
            self.log_file = None

        self.plateau_lr_metric = 0.0


        self.embed_regular_LT_dataset(self.data["train"])
        self.init_reference_dicts()
        self.specialize_embedding_dataset()

    def init_reference_dicts(self):

        self.label_frequencies = {}
        for label in self.final_labels:
            if label.item() not in self.label_frequencies:
                self.label_frequencies[label.item()] = 0
            self.label_frequencies[label.item()] += 1

        self.indices_by_label = {}
        for i, label in enumerate(self.final_labels):
            if label.item() not in self.indices_by_label:
                self.indices_by_label[label.item()] = []
            self.indices_by_label[label.item()].append(i)


        self.all_labels_text_clip = {}
        with torch.no_grad():
            for label in tqdm(range(1000)):
                self.all_labels_text_clip[label] = []
                templates = np.array(GENERIC_PROMPT_COLLECTIONS["ImageNet"])
                c = np.array(CLASSES)[label]
                texts = clip.tokenize([template.format(c) for template in templates]) 
                texts = texts.cuda()
                zeroshot_weights = self.text_model(texts).float()
                zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                    dim=-1, keepdim=True
                )
                self.all_labels_text_clip[label].append(zeroshot_weights)
            
        self.label_cats = {"many":[], "med":[], "few":[]}
        for label in set(self.final_labels):
            label = label.item()
            if self.label_frequencies[label] > 100:
                self.label_cats["many"].append(label)
            elif self.label_frequencies[label] > 20:
                self.label_cats["med"].append(label)
            else:
                self.label_cats["few"].append(label)

    def init_models(self, optimizer=True):
        # import pdb; pdb.set_trace()

        self.model_optim_params_list = []
        self.model_optim_params_list_LBFGS = []
        self.relu = nn.ReLU()

        print("Using", torch.cuda.device_count(), "GPUs.")

        # Initializing CLIP visual and Text models
        self.visual_model = torch.nn.DataParallel(self.clip_model.visual).cuda()
        text_model = TextEncoder(self.clip_model)
        self.text_model = torch.nn.DataParallel(text_model).cuda()

        # Initialize projection layer
        in_dim = self.model_opt["proj"]["params"]["in_dim"]
        out_dim = self.model_opt["proj"]["params"]["out_dim"]
        self.proj = torch.nn.DataParallel(
            nn.Linear(in_dim, out_dim, bias=False)
        ).cuda()

        if "model_dir" in self.config:
            print("Loading model weights from ", self.config["model_dir"])
            self.load_model(self.config["model_dir"])

        self.load_model()

        if self.training_opt["image_encoder_frozen"] is True:
            for param_name, param in self.visual_model.named_parameters():
                param.requires_grad = False

        if self.training_opt["text_encoder_frozen"] is True:
            for param_name, param in self.text_model.named_parameters():
                param.requires_grad = False

        optim_params_clip = self.model_opt["clip"]["optim_params"]
        self.model_optim_params_list.append(
            {
                "params": self.visual_model.parameters(),
                "lr": optim_params_clip["lr"],
                "momentum": optim_params_clip["momentum"],
                "weight_decay": optim_params_clip["weight_decay"],
            }
        )
        self.model_optim_params_list_LBFGS.extend(self.visual_model.parameters())

        self.model_optim_params_list.append(
            {
                "params": self.text_model.parameters(),
                "lr": optim_params_clip["lr"],
                "momentum": optim_params_clip["momentum"],
                "weight_decay": optim_params_clip["weight_decay"],
            }
        )
        self.model_optim_params_list_LBFGS.extend(self.text_model.parameters())

        optim_params_proj = self.model_opt["proj"]["optim_params"]
        self.model_optim_params_list.append(
            {
                "params": self.proj.parameters(),
                "lr": optim_params_proj["lr"],
                "momentum": optim_params_proj["momentum"],
                "weight_decay": optim_params_proj["weight_decay"],
            }
        )
        self.model_optim_params_list_LBFGS.extend(self.proj.parameters())

    def embed_regular_LT_dataset(self, data):

        self.final_images = []
        self.final_labels = []
        self.final_texts = []

        print("Embedding regular LT dataset.")
        with torch.no_grad():
            for im, label, _, path in tqdm(self.data["train"]):
                x = self.visual_model(im.half()).float()
                x = x / x.norm(dim=-1, keepdim=True)
                self.final_images.append(x)

                templates = np.array(GENERIC_PROMPT_COLLECTIONS["ImageNet"])[path.cpu()]
                classnames_for_labels = np.array(CLASSES)[label.cpu()]

                texts = clip.tokenize(t.format(c) for t,c in zip(templates, classnames_for_labels))
                texts = texts.cuda()
                zeroshot_weights = self.text_model(texts).float()
                zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                    dim=-1, keepdim=True
                )
                self.final_texts.append(zeroshot_weights)
                self.final_labels.append(label)

        self.final_images = torch.cat(self.final_images, dim=0)
        self.final_texts = torch.cat(self.final_texts, dim=0)
        self.final_labels = torch.cat(self.final_labels, dim=0)

    def specialize_embedding_dataset(self):

        var = check_config(self.config, "dataset_prep_variant")
        print("Specializing embedding dataset for variant:", var)

        self.upsampled_images = []
        self.upsampled_texts = []
        self.upsampled_labels = []

        if var == "1_per_class":
            self.balance_embedding_set(1, self.all_labels_text_clip)
        elif var == "400_per_class":
            self.balance_embedding_set(400, self.all_labels_text_clip)

    def balance_embedding_set(self, num_elements_per_class, text_dict=None):

        n = 1000*num_elements_per_class

        upsampled_images = torch.zeros((n, 1024))
        upsampled_texts = torch.zeros((n, 1024))
        upsampled_labels = []

        for i in range(n):
            label = random.choice(range(1000))

            idx = random.choice(self.indices_by_label[label])
            upsampled_images[i] = self.final_images[idx].cpu()

            idx_2 = random.choice(range(len(text_dict[label])))
            upsampled_texts[i] = text_dict[label][0][idx_2].cpu()

            upsampled_labels.append(label)

        self.upsampled_images = upsampled_images
        self.upsampled_texts = upsampled_texts
        self.upsampled_labels = upsampled_labels

        print("Upsampled images shape:", self.upsampled_images.shape)
        print("Upsampled texts shape:", self.upsampled_texts.shape)
        print("Upsampled labels shape:", len(self.upsampled_labels))


    def batch_forward(self, images, text, phase="train"):
        """
        This is a general single batch running function.
        """
        if phase == "train":
            self.logits = self.proj(text)


    def batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers

        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels):
        self.loss = 0

        if "PerformanceLoss" in self.criterions.keys():
            self.loss_perf = self.criterions["PerformanceLoss"](self.logits, labels)
            self.loss_perf *= self.criterion_weights["PerformanceLoss"]
            self.loss += self.loss_perf


    def shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def train_step(self, epoch, inputs, labels):

        if self.do_shuffle:
            inputs, labels = self.shuffle_batch(inputs, labels)
        inputs, labels = inputs.cuda(), labels.cuda()

        # If on training phase, enable gradients

        with torch.set_grad_enabled(True):

            # If training, forward with loss, and no top 5 accuracy calculation
            self.batch_forward(inputs, labels, phase="train")
            self.batch_loss(inputs)
            self.batch_backward()


    def train_step_LBFGS(self, epoch, inputs, labels):

        if self.do_shuffle:
            inputs, labels = self.shuffle_batch(inputs, labels)
        inputs, labels = inputs.cuda(), labels.cuda()

        # If on training phase, enable gradients

        def closure():
            self.model_optimizer.zero_grad()
            if self.criterion_optimizer:
                self.criterion_optimizer.zero_grad()

            self.batch_forward(inputs, labels, phase="train")
            self.batch_loss(inputs)
            # Back-propagation from loss outputs
            if self.loss.requires_grad:
                self.loss.backward()
            return self.loss

        self.model_optimizer.step(closure)
        if self.criterion_optimizer:
            self.criterion_optimizer.step(closure)

        # TODO track accuracies

    def train_epoch(self, epoch):

        torch.cuda.empty_cache()

        # Set model modes and set scheduler
        # In training, step optimizer scheduler and set model to train()
        if check_config(self.config, "plateaulr"):
            self.model_optimizer_scheduler.step(metrics=self.plateau_lr_metric)
        else:
            self.model_optimizer_scheduler.step()

        if self.criterion_optimizer:
            self.criterion_optimizer_scheduler.step()

        self.epoch_loss = 0.0

        def call_train_step(epoch, inputs, labels):
            if self.optimizer_variant == "LBFGS":
                self.train_step_LBFGS(epoch, inputs, labels)
            else:
                self.train_step(epoch, inputs, labels)

        call_train_step(epoch, self.final_images, self.final_texts)

        self.writer.add_scalar("Loss", self.loss, epoch)
        
        print("Config: {} Epoch: {} Loss: {}".format(self.training_opt["tf_folder"], epoch, self.loss))

        self.writer.add_scalar(
            "Learning Rate",
            float([group["lr"] for group in self.model_optimizer.param_groups][0]),
            epoch,
        )
        print("===> Saving checkpoint")
        self.save_latest(epoch)
        # self.save_model(epoch)

    def train(self):
        # When training the network
        print_str = ["Phase: train"]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        print_write(["Do shuffle??? --- ", self.do_shuffle], self.log_file)

        # Initialize best model
        self.best_model_weights = {}
        self.best_model_weights["visual_model"] = copy.deepcopy(
            self.visual_model.state_dict()
        )
        self.best_model_weights["text_model"] = copy.deepcopy(
            self.text_model.state_dict()
        )
        # if self.training_opt["phaseA"] is not True:
        self.best_model_weights["proj"] = copy.deepcopy(self.proj.state_dict())


        end_epoch = self.training_opt["num_epochs"]

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):

            self.train_epoch(epoch)

        print()
        print("Training Complete.")

        print("Final loss: ", self.loss)

        print("Test set seperability: ")
        self.eval(self.data["test"])

        print("Done")

    def eval(self, dataset):

        with torch.no_grad():

            test_images = []
            test_labels = []

            for im, label, _, path in tqdm(dataset):
                x = self.visual_model(im.half()).float()
                x = x / x.norm(dim=-1, keepdim=True)
                test_images.append(x)
                test_labels.append(label)

            test_images = torch.cat(test_images, dim=0)
            test_labels = torch.cat(test_labels, dim=0)

            many_test_images, med_test_images, few_test_images = [], [], []
            many_test_labels, med_test_labels, few_test_labels = [], [], []

            for i,l in tqdm(zip(test_images, test_labels)):
                
                if l.item() in self.label_cats["many"]:
                    many_test_images.append(i)
                    many_test_labels.append(l.item())
                elif l.item() in self.label_cats["med"]:
                    med_test_images.append(i)
                    med_test_labels.append(l.item())
                elif l.item() in self.label_cats["few"]:
                    few_test_images.append(i)
                    few_test_labels.append(l.item())

            many_test_images = torch.stack(many_test_images,dim=1)
            med_test_images = torch.stack(med_test_images,dim=1)
            few_test_images = torch.stack(few_test_images,dim=1)
            many_test_labels = torch.tensor(many_test_labels)
            med_test_labels = torch.tensor(med_test_labels)
            few_test_labels = torch.tensor(few_test_labels)

            data_proj = self.proj(self.final_texts)
            clf_proj = LogisticRegression(random_state=0, verbose=10, n_jobs=-1).fit(data_proj.cpu(), self.final_labels.cpu())

            print("Many:", clf_proj.score(many_test_images.T.cpu(), many_test_labels.cpu()))
            print("Med:", clf_proj.score(med_test_images.T.cpu(), med_test_labels.cpu()))
            print("Few:", clf_proj.score(few_test_images.T.cpu(), few_test_labels.cpu()))
            print("All:", clf_proj.score(test_images.cpu(), test_labels.cpu()))


    def load_model(self, model_dir=None):
        model_dir = self.training_opt["log_dir"] if model_dir is None else model_dir

        if os.path.isfile(model_dir + "/final_model_checkpoint.pth"):

            model_dir += "/final_model_checkpoint.pth"
            checkpoint = torch.load(model_dir, map_location="cpu")
            model_state = checkpoint["state_dict_best"]
            epoch = checkpoint["epoch"]
            print(f"Loading best model which was trained for {epoch} epochs")

        elif os.path.isfile(model_dir + "/latest_model_checkpoint.pth"):

            model_dir += "/latest_model_checkpoint.pth"
            checkpoint = torch.load(model_dir, map_location="cpu")
            model_state = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            print(f"Training hasn't finished, loading model trained for {epoch} epochs")
        else:
            print("No pretrained model")
            return

        print("Loading model from %s" % (model_dir))

        # checkpoint = torch.load(model_dir, map_location="cpu")
        # model_state = checkpoint["state_dict_best"]

        self.visual_model.load_state_dict(model_state["visual_model"])
        self.text_model.load_state_dict(model_state["text_model"])
        self.proj.load_state_dict(model_state["proj"])

        # if self.test_mode is True:
        #     self.adapter.load_state_dict(model_state["classifier"])

    def save_latest(self, epoch):
        model_weights = {}
        model_weights["visual_model"] = copy.deepcopy(self.visual_model.state_dict())
        model_weights["text_model"] = copy.deepcopy(self.text_model.state_dict())
        # if self.training_opt["phaseA"] is not True:
        model_weights["proj"] = copy.deepcopy(self.proj.state_dict())

        model_states = {"epoch": epoch, "state_dict": model_weights}

        model_dir = os.path.join(
            self.training_opt["log_dir"], "latest_model_checkpoint.pth"
        )
        torch.save(model_states, model_dir)

    # def save_model(
    #     self, epoch, best_epoch, best_model_weights, best_acc, centroids=None
    # ):

    #     model_states = {
    #         "epoch": epoch,
    #         "best_epoch": best_epoch,
    #         "state_dict_best": best_model_weights,
    #         "best_acc": best_acc,
    #         "centroids": centroids,
    #     }

    #     model_dir = os.path.join(
    #         self.training_opt["log_dir"], "final_model_checkpoint.pth"
    #     )

    #     torch.save(model_states, model_dir)
