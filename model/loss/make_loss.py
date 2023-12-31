# encoding: utf-8
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .arcface import ArcFace
from .cmpm import CMPMLoss
from .cfa import CFA
from .id_loss import Id_Loss
from einops import rearrange

def make_loss(cfg, num_classes):    # modified by gu
    loss = cfg.MODEL.LOSS_TYPE
    feat_dim = cfg.MODEL.FINAL_FEARTURE_DIM #cfg.MODEL.FEARTURE_DIM
    arcface_loss = ArcFace(s=cfg.MODEL.ARC_SCALE, m=cfg.MODEL.ARC_MARGIN) #CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    i2t_loss = CMPMLoss(epsilon=1e-8, num_classes=num_classes, feature_size=feat_dim)
    cfa_loss = CFA(s=cfg.MODEL.ARC_SCALE, m=cfg.MODEL.ARC_MARGIN)
    # xent = torch.nn.CrossEntropyLoss(reduction='mean')
    # id_loss = Id_Loss(xent)
    if 'triplet' in cfg.MODEL.LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        id_loss = Id_Loss(xent)
        print("label smooth on, num classes:", num_classes)

    if loss == 'softmax_idloss':
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            id_loss = Id_Loss(xent)
        else:
            import torch.nn as nn
            xent = nn.CrossEntropyLoss(reduction='mean')
            id_loss = Id_Loss(xent)

        def loss_func(i_score, t_score, feat_g, feat_q, labels):
            ID_LOSS = id_loss(i_score, t_score, labels)
            METRIC_LOSS = i2t_loss(feat_g, feat_q, labels)
            return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                   cfg.MODEL.METRIC_LOSS_WEIGHT * METRIC_LOSS
    elif loss == 'cmpm':
        def loss_func(feat_g, feat_q, labels):
            losses = {}
            cmpm_loss , _ = i2t_loss(feat_g, feat_q, labels)
            cmpm_loss = cfg.MODEL.METRIC_LOSS_WEIGHT * cmpm_loss
            LOSS = {'loss': cmpm_loss}
            losses.update(LOSS)
            return losses
    elif loss == 'attr':
        def loss_func(feat_g, feat_q, feat_attr, labels, label_attr):
            losses = {}
            # cmpm_loss, _ = i2t_loss(feat_g, feat_q, labels)
            # cmpm_loss = cfg.MODEL.METRIC_LOSS_WEIGHT * cmpm_loss
            # LOSS = {'loss': cmpm_loss}
            # losses.update(LOSS)
            attr_loss = bce_loss(feat_attr, label_attr)
            attr_loss = cfg.MODEL.AUX_LOSS_WEIGHT * attr_loss
            LOSS2 = {'aux_loss': attr_loss}
            losses.update(LOSS2)
            return losses
    elif loss == 'cmpm+attr':
        def loss_func(feat_g, feat_q, feat_attr, labels, label_attr):
            losses = {}
            cmpm_loss, _ = i2t_loss(feat_g, feat_q, labels)
            cmpm_loss = cfg.MODEL.METRIC_LOSS_WEIGHT * cmpm_loss
            LOSS = {'loss': cmpm_loss}
            losses.update(LOSS)
            attr_loss = bce_loss(feat_attr, label_attr)
            attr_loss = cfg.MODEL.AUX_LOSS_WEIGHT * attr_loss
            LOSS2 = {'aux_loss': attr_loss}
            losses.update(LOSS2)
            return losses
    elif loss == 'cmpm+arc':
        def loss_func(feat_g, feat_q, labels):
            losses = {}
            cmpm_loss, _ = i2t_loss(feat_g, feat_q, labels)
            cmpm_loss = cfg.MODEL.METRIC_LOSS_WEIGHT * cmpm_loss
            LOSS = {'loss': cmpm_loss}
            losses.update(LOSS)
            aux_loss = arcface_loss(feat_g, feat_q, labels)
            aux_loss = cfg.MODEL.AUX_LOSS_WEIGHT * aux_loss
            LOSS = {'aux_loss': aux_loss}
            losses.update(LOSS)
            return losses
    elif loss == 'arc':
        def loss_func(feat_g, feat_q, labels):
            losses = {}
            arc_loss = arcface_loss(feat_g, feat_q, labels)
            arc_loss = cfg.MODEL.METRIC_LOSS_WEIGHT * arc_loss
            LOSS = {'loss': arc_loss}
            losses.update(LOSS)
            return losses
    elif loss == 'cfa':
        def loss_func(feat_g, feat_q, labels):
            losses = {}
            c_loss = cfa_loss(feat_g, feat_q, labels)
            c_loss = cfg.MODEL.METRIC_LOSS_WEIGHT * c_loss
            LOSS = {'loss': c_loss}
            losses.update(LOSS)
            return losses
    elif loss == 'arc+id':
        def loss_func(feat_g, feat_q, i_score, t_score, labels):
            losses = {}
            arc_loss = arcface_loss(feat_g, feat_q, labels)
            arc_loss = cfg.MODEL.METRIC_LOSS_WEIGHT * arc_loss
            LOSS = {'loss': arc_loss}
            losses.update(LOSS)
            aux_loss = id_loss(i_score, t_score, labels)
            aux_loss = cfg.MODEL.AUX_LOSS_WEIGHT * aux_loss
            LOSS = {'aux_loss': aux_loss}
            losses.update(LOSS)
            return losses
    elif loss == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected loss should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(loss))
    return loss_func, None


