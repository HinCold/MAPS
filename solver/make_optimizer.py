import torch

def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        print("optimizer is：Adam")
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center

def optimizer_function(cfg, model):
    if cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        #
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR, betas=(cfg.SOLVER.ALPHA, cfg.SOLVER.BETA), eps=cfg.SOLVER.EPS)
        # print("optimizer is：Adam")
        # optimizer_attr = torch.optim.Adam(model.fc_attr.parameters(), lr=0.0001, betas=(0.9, 0.999),weight_decay=0.0005)

        # special_layers = torch.nn.ModuleList([model.attr_branch, model.fc_attr1])
        # special_layers = torch.nn.ModuleList([model.text_backbone, model.shared_block, model.cm_branch])
        # # special_layers = torch.nn.ModuleList([model.text_backbone, model.cm_branch])
        # # special_layers = torch.nn.ModuleList([model.shared_block, model.cm_branch])
        # special_layers_params = list(map(id, special_layers.parameters()))
        # # print(special_layers_params)
        # base_params = filter(lambda p: id(p) not in special_layers_params, model.parameters())
        #
        # optimizer = torch.optim.Adam([{'params': base_params},
        #                          {'params': special_layers.parameters(), 'lr': 0.0000}], lr=cfg.SOLVER.BASE_LR,
        #                              betas=(cfg.SOLVER.ALPHA, cfg.SOLVER.BETA), eps=cfg.SOLVER.EPS)
        # optimizer = torch.optim.Adam(special_layers.parameters(), lr=cfg.SOLVER.BASE_LR,
        #                          betas=(cfg.SOLVER.ALPHA, cfg.SOLVER.BETA), eps=cfg.SOLVER.EPS)


    return optimizer,  None