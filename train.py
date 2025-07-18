"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import pipeline.tasks as tasks
from pipeline.common.config import Config
from pipeline.common.dist_utils import get_rank, init_distributed_mode
from pipeline.common.logger import setup_logger
from pipeline.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from pipeline.common.registry import registry
from pipeline.common.utils import now

# imports modules for registration
from pipeline.datasets.builders import *
from pipeline.models import *
from pipeline.processors import *
from pipeline.runners import *
from pipeline.tasks import *

import warnings
warnings.filterwarnings('ignore', message='Precision and F-score are ill-defined*')
warnings.filterwarnings('ignore', message='Recall and F-score are ill-defined*')

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--cfg-path", 
        type=str, 
        default="/home/zhaoyang/project/drug-drug-interaction/train_configs/drugchat.yaml", 
        help="path to configuration file."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
   
    if get_rank() == 0:
        cfg_dict = cfg.to_dict()
        wandb_run_name = cfg_dict['run']['output_dir'].split('/')[-1]
        setup_logger()
        wandb.init(project="drugchat", config=cfg_dict, name=wandb_run_name, job_type="training")
        
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    
    if get_rank() == 0:
        wandb.watch(model, log="all", log_freq=5000)
        
    runner.train()
    
    if get_rank() == 0:
        wandb.finish()
        

if __name__ == "__main__":
    main()
