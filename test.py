# -*- coding: utf-8 -*-
"""
@Time ： 21-10-6 下午11:33
@Auth ： Nathan
@File ：test.py.py
@WF ： ...
"""
import torch
import numpy as np
import random
import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from tasks.processor import do_inference, do_show
from utils.logger import setup_logger

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--c", default="/home/lh/project/TTIPS/new_logs/fusion+msfm+cfa_gpu3/config.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument(
        "--show", action="store_true", help="show result."
    )
    args = parser.parse_args()

    set_seed(cfg.SOLVER.SEED)

    if args.c != "":
        cfg.merge_from_file(args.c)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("text-basedPS.train", output_dir, if_train=False)
    logger.info(args)

    if args.c != "":
        logger.info("Loaded configuration file {}".format(args.c))
        with open(args.c, 'r') as cf:
            config_str = "\n" + cf.read()
            # logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_classes = make_dataloader(cfg)

    if args.show:
        do_show(cfg, train_loader_normal)
    else:
        model = make_model(cfg, num_class=num_classes)
        model.load_param(cfg.TEST.WEIGHT)

        do_inference(cfg,
                     model,
                     train_loader_normal)




