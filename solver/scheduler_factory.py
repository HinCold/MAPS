""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""

from bisect import bisect_right
from .cosine_lr import CosineLRScheduler
import torch

def create_CosineLRscheduler(cfg, optimizer):
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    # type 1
    # lr_min = 0.01 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
    # type 2
    lr_min = 0.002 * cfg.SOLVER.BASE_LR
    warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
    # type 3
    # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    print('WARMUP_EPOCHS', warmup_t)
    noise_range = None

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler

def lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.LR_TYPE == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(cfg.SOLVER.STEPS), gamma=cfg.SOLVER.GAMMA)
        print("lr_scheduler is StepLR")

    elif cfg.SOLVER.LR_TYPE == 'CosineLR':
        scheduler = create_CosineLRscheduler(cfg, optimizer)
        print("lr_scheduler is CosineLR")

    elif cfg.SOLVER.LR_TYPE == 'MultiStepLR':
        print("lr_scheduler is MultiStepLR")
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, milestones=cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA)

    elif cfg.SOLVER.LR_TYPE == 'CosineAnnealingLR':
        print("lr_scheduler is CosineAnnealingLR")
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0000007)
    return scheduler

def gradual_warmup(epoch,init_lr,optimizer,epochs):
    lr = init_lr
    if epoch < epochs:
        warmup_percent_done = (epoch+1) / epochs
        warmup_learning_rate = init_lr * warmup_percent_done
        lr = warmup_learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]