#!/usr/bin/env python

import shutil
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from model.loss import make_loss
from solver import make_optimizer, optimizer_function
from solver.scheduler_factory import lr_scheduler
from tasks.processor import do_train
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--c", default="configs/pedes/maps_cfa_new.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--fineturn", action="store_true", help="show result."
    )
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.c != "":
        cfg.merge_from_file(args.c)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("text-basedPS.train", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    shutil.copy('/home/lh/project/TTIPS/model/make_model.py', cfg.OUTPUT_DIR+'/make_model.py')
    shutil.copy('/home/lh/project/TTIPS/model/backbone/image/ibn_resnet50.py', cfg.OUTPUT_DIR + '/ibn_resnet50.py')
    shutil.copy('/home/lh/project/TTIPS/model/backbone/text/bert.py', cfg.OUTPUT_DIR + '/bert.py')
    shutil.copy(args.c, cfg.OUTPUT_DIR+'/config.yml')
    #logger.info(args)

    # if args.config_file != "":
    #     logger.info("Loaded configuration file {}".format(args.config_file))
    #     with open(args.config_file, 'r') as cf:
    #         config_str = "\n" + cf.read()
    #         logger.info(config_str)
    #logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_classes = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes).cuda()
    if args.fineturn:
        model.load_param(cfg.TEST.WEIGHT)
    # special_layers = torch.nn.ModuleList([model.fc_attr])
    # compute the model size:
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # print(model)
    loss_func, attr_bce_loss = make_loss(cfg, num_classes=num_classes)

    optimizer, aux_optimizer = optimizer_function(cfg, model)
    #
    scheduler = lr_scheduler(cfg, optimizer)
    #
    do_train(
        cfg,
        model,
        #params,
        train_loader,
        train_loader_normal, #test
        optimizer,
        aux_optimizer,
        scheduler,
        loss_func,
        args.local_rank
    )

