
import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, accuracy
from torch.cuda import amp
import torch.distributed as dist
from solver.scheduler_factory import gradual_warmup

def do_train(cfg,
             model,
             #params,
             train_loader,
             val_loader,
             optimizer,
             aux_optimizer,
             scheduler,
             loss_fn,
             local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("text-basedPS.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    loss_aux_meter = AverageMeter()
    top1 = AverageMeter()
    img_acc_meter = AverageMeter()
    text_acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(cfg.TEST.IMS_PER_BATCH * len(val_loader), cfg.MODEL.FINAL_FEARTURE_DIM, cfg.DATASETS.NAMES)
    scaler = amp.GradScaler()
    aux_params_update_every = 1

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_aux_meter.reset()
        evaluator.reset()
        img_acc_meter.reset()
        text_acc_meter.reset()
        model.train()
        scheduler.step(epoch)
        # if epoch < cfg.SOLVER.WARMUP_EPOCHS:
        #     print('learning rate warm_up, epochs:', epoch)
        #     optimizer = gradual_warmup(epoch, cfg.SOLVER.BASE_LR, optimizer, epochs=cfg.SOLVER.WARMUP_EPOCHS)

        for n_iter, (img, caption, mask, labels, _, label_attr) in enumerate(train_loader):
            optimizer.zero_grad()
            # optimizer_attr.zero_grad()
            img = img.to(device)
            caption = caption.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            # if label_attr is not None:
            # label_attr = label_attr.to(device)
            with amp.autocast(enabled=True):
                if "id" in cfg.MODEL.LOSS_TYPE:
                    feat_g, feat_q, labels, img_cls_score, text_cls_score = model(img, caption, mask, labels)
                    loss_dict = loss_fn(feat_g, feat_q, img_cls_score, text_cls_score, labels)
                else:
                    feat_g, feat_q, labels = model(img, caption, mask, labels)

                    loss_dict = loss_fn(feat_g, feat_q, labels)
                # loss_dict = loss_fn(feat_g, feat_q, feat_attr, labels, label_attr)

            losses = sum(loss for loss in loss_dict.values())

            # optimizer.zero_grad()
            # losses.backward()
            # optimizer.step()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            # print(img_cls_score, text_cls_score)
            # print(img_acc, text_acc)
            # c
            loss_meter.update(loss_dict['loss'].item(), img.shape[0])

            # loss_aux_meter.update(loss_dict['aux_loss'].item(), img.shape[0])
            # img_acc = (img_cls_score.max(1)[1] == labels).float().mean()
            # text_acc = (text_cls_score.max(1)[1] == labels).float().mean()
            # img_acc_meter.update(img_acc, 1)
            # text_acc_meter.update(text_acc, 1)
            # # torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] loss : {:.5f}, aux loss: {:.3f}, Current Lr: {:.2e}, len: {:.1f}, img Acc: {:.3f},"
                            "text Acc: {:.3f},"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, loss_aux_meter.avg, optimizer.param_groups[0]['lr'], len(labels), img_acc_meter.avg,
                                    text_acc_meter.avg ) )
                # print(len(labels), feat_g.shape, feat_q.shape)
                # c

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)



        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0 \
            and epoch > cfg.SOLVER.MAX_EPOCHS - 10:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, caption, mask, labels, img_path) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            caption = caption.to(device)
                            labels = labels.to(device)
                            mask = mask.to(device)
                            image_embeddings, text_embeddings = model(img, caption, mask)
                            evaluator.update((text_embeddings, labels, image_embeddings, labels, img_path))
                    cmc, mAP = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                #eval
                model.eval()
                for n_iter, (img, caption, mask, labels, img_path, label_attr) in enumerate(val_loader):
                    # print('n_iter', n_iter)
                    with torch.no_grad():
                        img = img.to(device)
                        caption = caption.to(device)
                        labels = labels.to(device)
                        mask = mask.to(device)
                        feat_g, feat_q = model(img, caption, mask)
 
                        evaluator.update((feat_q, labels, feat_g, labels, img_path), img.shape[0])
                # evaluator.save_feat()
                t2i_cmc, t2i_mAP, i2t_cmc, i2t_mAP = evaluator.compute()
                logger.info("Validation Results ")
                logger.info("t2i_mAP: {:.1%}".format(t2i_mAP))
                logger.info("i2t_mAP: {:.1%}".format(i2t_mAP))
                for r in [1, 5, 10]:
                    logger.info("t2i_cmc curve, Rank-{:<3}:{:.2%}".format(r, t2i_cmc[r - 1]))
                    logger.info("i2t_cmc curve, Rank-{:<3}:{:.2%}".format(r, i2t_cmc[r - 1]))
                # logger.info("Runtime Results ")
                # logger.info("Avg Time :{}; FPS :{}".format(fps_meter.avg, 1/(fps_meter.avg)))

                # torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 test_loader):
    device = "cuda"
    logger = logging.getLogger("text-basedPS.train")
    logger.info("Enter inferencing")

    # evaluator = Attr_R1_mAP_eval(cfg.SOLVER.IMS_PER_BATCH * len(test_loader), cfg.MODEL.FINAL_FEARTURE_DIM)
    evaluator = R1_mAP_eval(cfg.SOLVER.IMS_PER_BATCH * len(test_loader), cfg.MODEL.FINAL_FEARTURE_DIM, dataset=cfg.DATASETS.NAMES)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    fps_meter= AverageMeter()
    fps_meter.reset()
    for n_iter, (img, caption, mask, labels, img_path, label_attr) in enumerate(test_loader):
        with torch.no_grad():
            img = img.to(device)
            caption = caption.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            start_time = time.time()
            feat_g, feat_q  = model(img, caption, mask)
            fps_meter.update(time.time()-start_time, len(img))
            evaluator.update((feat_q, labels, feat_g, labels, img_path), img.shape[0])
    evaluator.save_feat()
    t2i_cmc, t2i_mAP, i2t_cmc, i2t_mAP = evaluator.compute()

    logger.info("Validation Results ")
    logger.info("t2i_mAP: {:.1%}".format(t2i_mAP))
    logger.info("i2t_mAP: {:.1%}".format(i2t_mAP))
    for r in [1, 5, 10]:
        logger.info("t2i_cmc curve, Rank-{:<3}:{:.2%}".format(r, t2i_cmc[r - 1]))
        logger.info("i2t_cmc curve, Rank-{:<3}:{:.2%}".format(r, i2t_cmc[r - 1]))
    logger.info("Runtime Results ")
    logger.info("Avg Time :{}; FPS :{}".format(fps_meter.avg, 1/(fps_meter.avg)))

    return t2i_cmc[0], t2i_cmc[4]

def do_show(cfg, test_loader):
    device = "cuda"
    logger = logging.getLogger("text-basedPS.train")
    logger.info("Enter inferencing")

    # evaluator = Attr_R1_mAP_eval(cfg.SOLVER.IMS_PER_BATCH * len(test_loader), cfg.MODEL.FINAL_FEARTURE_DIM)
    evaluator = R1_mAP_eval(cfg.SOLVER.IMS_PER_BATCH * len(test_loader), cfg.MODEL.FINAL_FEARTURE_DIM)

    evaluator.reset()

    # if device:
    #     if torch.cuda.device_count() > 1:
    #         print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
    #         model = nn.DataParallel(model)
    #     model.to(device)

    evaluator.visualize()


