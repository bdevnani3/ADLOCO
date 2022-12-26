import torch
import torch.optim as optim

from algorithms import *
from diffgrad import diffgrad
from utils import *


def init_criterions(self):
    criterion_defs = self.config["criterions"]
    self.criterions = {}
    self.criterion_weights = {}

    for key, val in criterion_defs.items():
        def_file = val["def_file"]
        loss_args = list(val["loss_params"].values())

        self.criterions[key] = source_import(def_file).create_loss(*loss_args).cuda()
        self.criterion_weights[key] = val["weight"]

        if val["optim_params"]:
            print("Initializing criterion optimizer.")
            optim_params = val["optim_params"]
            optim_params = [
                {
                    "params": self.criterions[key].parameters(),
                    "lr": optim_params["lr"],
                    "momentum": optim_params["momentum"],
                    "weight_decay": optim_params["weight_decay"],
                }
            ]
            # Initialize criterion optimizer and scheduler
            (
                self.criterion_optimizer,
                self.criterion_optimizer_scheduler,
            ) = self.init_optimizers(optim_params)
        else:
            self.criterion_optimizer = None


def init_optimizers(self, optim_params):

    # Optimizers

    if self.optimizer_variant == "LBFGS":
        optim_params = self.config["optimizer_args"]
        optimizer = optim.LBFGS(
            self.model_optim_params_list_LBFGS,
            history_size=int(optim_params["history_size"]),
            max_iter=int(optim_params["max_iter"]),
            lr=float(optim_params["lr"]),
        )
    elif self.optimizer_variant == "Adam":
        optimizer = optim.Adam(optim_params)
    elif self.optimizer_variant == "diffgrad":
        optimizer = diffgrad(optim_params)
    else:
        optimizer = optim.SGD(optim_params)

    # Schedulers

    if self.config["coslr"]:
        print("===> Using coslr eta_min={}".format(self.config["endlr"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.training_opt["num_epochs"], eta_min=self.config["endlr"]
        )
    elif check_config(self.config, "plateaulr"):
        print("===> Using plateaulr ")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=5,
            verbose=True,
        )
    elif check_config(self.config, "multisteplr"):
        print("===> Using multisteplr ")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config["milestones"],
            gamma=self.config["gamma"],
            verbose=True,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_params["step_size"],
            gamma=self.scheduler_params["gamma"],
        )
    return optimizer, scheduler
