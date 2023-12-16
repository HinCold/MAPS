import torch
import numpy as np
import torch.nn.functional as F
import os
import cv2
from utils.reranking import re_ranking
from PIL import Image
from utils.iotools import mkdir_if_missing
import shutil
import pandas as pd
from haishoku.haishoku import Haishoku
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import time
def accuracy(output, target):
    batch_size = target.size(0)
    attr_num = target.size(1)

    output = torch.sigmoid(output).cpu().numpy()
    output = np.where(output > 0.5, 1, 0)
    pred = torch.from_numpy(output).long()
    target = target.cpu().long()
    correct = pred.eq(target)
    correct = correct.numpy()

    res = []
    for k in range(attr_num):
        res.append(1.0*sum(correct[:,k]) / batch_size)
    return sum(res) / attr_num



def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty

        cmc[0] = -1
        return ap, cmc
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1

    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def evaluate(qf, ql, gf, gl):
    # print('qf, ql, gf, gl', qf.shape, ql.shape, gf.shape, gl.shape)
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    # score = cost_matrix_cosine(qf, gf)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    index = np.argsort(score)
    index = index[::-1]


    gl=gl.cuda().data.cpu().numpy()
    ql=ql.cuda().data.cpu().numpy()
    query_index = np.argwhere(gl == ql)
    ap, cmc = compute_mAP(index, query_index)
    return ap, cmc


def evaluatevis(qf, ql, gf, gl):
    # print('qf, ql, gf, gl', qf.shape, ql.shape, gf.shape, gl.shape)
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    # score = cost_matrix_cosine(qf, gf)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    index = np.argsort(score)
    index = index[::-1]


    gl=gl.cuda().data.cpu().numpy()
    ql=ql.cuda().data.cpu().numpy()
    query_index = np.argwhere(gl == ql)
    ap, cmc = compute_mAP(index, query_index)
    return ap, cmc, index, score

def evaluate_rerank(ql, gl, pre_index):
    index = pre_index

    gl=gl.cuda().data.cpu().numpy()
    ql=ql.cuda().data.cpu().numpy()
    query_index = np.argwhere(gl == ql)
    CMC_tmp = compute_mAP(index, query_index)
    return CMC_tmp

class R1_mAP_eval():
    def __init__(self, max_size, feature_size, dataset='pedes'):
        super(R1_mAP_eval, self).__init__()
        self.query_features = torch.zeros((max_size, feature_size)).cuda()
        self.query_labels = torch.zeros(max_size).cuda()
        # we input the two times of images, so we need to select half of them
        self.gallery_features = torch.zeros((max_size, feature_size)).cuda()
        self.gallery_labels = torch.zeros(max_size).cuda()
        self.img_paths = []
        self.feature_size = feature_size
        self.max_size = max_size
        self.index = 0
        self.dataset = dataset
        print('using dataset ', self.dataset)
        self.top10 = {}

    def reset(self):
        self.query_features = torch.zeros((self.max_size, self.feature_size)).cuda()
        self.query_labels = torch.zeros(self.max_size).cuda()
        # we input the two times of images, so we need to select half of them
        self.gallery_features = torch.zeros((self.max_size, self.feature_size)).cuda()
        self.gallery_labels = torch.zeros(self.max_size).cuda()

        self.index = 0

    def update(self, output, interval):  # called once for each batch
        text_embeddings, labels, image_embeddings, labels, img_path, = output
        self.gallery_features[self.index: self.index + interval] = image_embeddings
        self.query_features[self.index: self.index + interval] = text_embeddings
        self.gallery_labels[self.index:self.index + interval] = labels
        self.query_labels[self.index:self.index + interval] = labels

        self.index = self.index + interval

        self.img_paths.extend(img_path)

    def compute(self):  # called after each epoch
        # print('self.index ', self.index )
        self.gallery_features = self.gallery_features[:self.index] # image
        self.query_features = self.query_features[:self.index] # text
        self.gallery_labels = self.gallery_labels[:self.index]
        self.query_labels = self.query_labels[:self.index]
        self.img_paths = self.img_paths[:self.index]

        if self.dataset == 'pedes':
            self.img_paths = self.img_paths[::2]
            self.gallery_features = self.gallery_features[::2]
            self.gallery_labels = self.gallery_labels[::2]
        if self.dataset == 'f30k':
            self.img_paths = self.img_paths[::5]
            self.gallery_features = self.gallery_features[::5]
            self.gallery_labels = self.gallery_labels[::5]
        if self.dataset == 'cub':
            print('this is ', self.dataset)
            self.img_paths = self.img_paths[::10]
            self.gallery_features = self.gallery_features[::10]
            self.gallery_labels = self.gallery_labels[::10]
        self.query_features = self.query_features / (self.query_features.norm(dim=1, keepdim=True) + 1e-12)
        self.gallery_features = self.gallery_features / (self.gallery_features.norm(dim=1, keepdim=True) + 1e-12)

        t2i_CMC = torch.IntTensor(len(self.gallery_labels)).zero_()
        t2i_ap = 0.0
        for i in range(len(self.query_labels)):
            t2i_ap_tmp, t2i_CMC_tmp = evaluate(self.query_features[i], self.query_labels[i], self.gallery_features, self.gallery_labels)
            if t2i_CMC_tmp[0] == -1:
                continue
            t2i_CMC = t2i_CMC + t2i_CMC_tmp
            t2i_ap += t2i_ap_tmp

        t2i_CMC = t2i_CMC.float()
        t2i_CMC = t2i_CMC / len(self.query_labels)


        i2t_CMC = torch.IntTensor(len(self.query_labels)).zero_()
        i2t_ap = 0.0
        for i in range(len(self.gallery_labels)):
            i2t_ap_tmp, i2t_CMC_tmp = evaluate(self.gallery_features[i], self.gallery_labels[i], self.query_features, self.query_labels)
            if i2t_CMC_tmp[0] == -1:
                continue
            i2t_CMC = i2t_CMC + i2t_CMC_tmp
            i2t_ap += i2t_ap_tmp

        i2t_CMC = i2t_CMC.float()
        i2t_CMC = i2t_CMC / len(self.gallery_labels)

        return t2i_CMC, t2i_ap / len(self.query_labels), i2t_CMC, i2t_ap / len(self.gallery_labels) #ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP

    def save_feat(self):
                # self.query_labels = self.query_labels[:self.index]

        gallery_features = self.gallery_features[:self.index]
        query_features = self.query_features[:self.index]
        gallery_labels = self.gallery_labels[:self.index]
        query_labels = self.query_labels[:self.index]
        img_paths = self.img_paths[:self.index]

        img_paths = img_paths[::2]
        gallery_features = gallery_features[::2]
        gallery_labels = gallery_labels[::2]
        query_features = query_features / (query_features.norm(dim=1, keepdim=True) + 1e-12)
        gallery_features = gallery_features / (gallery_features.norm(dim=1, keepdim=True) + 1e-12)
        
        print('len(img_paths), len(gallery_features)', len(img_paths), len(gallery_features))
        # idx = self.query_labels<20
        # y = self.query_labels[idx]
        # print(y)
        np.save('/home/lh/project/TTIPS/datasets/data/cfa/gallery_f.npy', gallery_features.cpu())
        np.save('/home/lh/project/TTIPS/datasets/data/cfa/query_f.npy',  query_features.cpu())
        np.save('/home/lh/project/TTIPS/datasets/data/cfa/gallery_labels.npy',  gallery_labels.cpu())
        np.save('/home/lh/project/TTIPS/datasets/data/cfa/query_labels.npy', query_labels.cpu())
        np.save('/home/lh/project/TTIPS/datasets/data/cfa/img_paths.npy', img_paths)

    def visualize(self):
        self.query_features = torch.from_numpy(
            np.load('/home/lh/project/TTIPS/datasets/data/cfa/query_f.npy')).cuda()
        self.gallery_features = torch.from_numpy(
            np.load('/home/lh/project/TTIPS/datasets/data/cfa/gallery_f.npy')).cuda()
        self.query_labels = torch.from_numpy(
            np.load('/home/lh/project/TTIPS/datasets/data/cfa/query_labels.npy')).cuda()
        self.gallery_labels = torch.from_numpy(
            np.load('/home/lh/project/TTIPS/datasets/data/cfa/gallery_labels.npy')).cuda()
        self.img_paths = np.load('/home/lh/project/TTIPS/datasets/data/cfa/img_paths.npy')


        txt_path = '/home/lh/project/TTIPS/datasets/data/BERT_encode/pedes/test.csv'
        csv_data = pd.read_csv(txt_path, header=None)
        captions = csv_data[2]
        GT = csv_data[1]
        # print(captions)
        cnt = 0
        for i in range(len(self.query_labels)):
            # if self.query_labels[i].cpu().int().item() != 18:
            #     continue
            query = self.query_features[i].view(-1, 1)
            scores = torch.mm(self.gallery_features, query)
            # scores = torch.cosine_similarity(self.query_features[i], self.gallery_features, dim=0)
            scores = scores.squeeze(1).cpu()
            scores = scores.numpy()

            index = np.argsort(scores)
            index = index[::-1]

            text = captions[i]
            GT_img = GT[i]
            notin_flag = True
            for iii in index[:5]:
                if self.query_labels[i].cpu().int().item() == self.gallery_labels[iii].cpu().int().item():
                    notin_flag = False
            # return a
            if notin_flag:
                mkdir_if_missing('/home/lh/project/TTIPS/vis_fail/')
                tmp_path = '/home/lh/project/TTIPS/vis_fail/query_'+str(self.query_labels[i].cpu().int().item())
            else:
                mkdir_if_missing('/home/lh/project/TTIPS/vis_success/')
                tmp_path = '/home/lh/project/TTIPS/vis_success/query_'+str(self.query_labels[i].cpu().int().item())

            if os.path.exists(tmp_path):
                continue
            else:
                mkdir_if_missing(tmp_path)
            name = 'GT_'+'id_'+str(self.query_labels[i].cpu().int().item())+'_'+GT_img.split('/')[-1]
            im = cv2.imread( '/data/lh/datasets/person_search/CUHK-PEDES/imgs/' + GT_img)
            im = cv2.resize(im, (128, 384))
            font = cv2.FONT_HERSHEY_SIMPLEX
            im = cv2.putText(im, 'GT', (0, 20), font, 0.6, (250, 228, 199), 2)   
            # print('score[idx]', scores[idx])
            path = tmp_path+'/'+name
            cv2.imwrite(path, im)

            with open(tmp_path+'/id_'+str(self.query_labels[i].cpu().int().item())+"_caption.txt", "w") as f:
                f.write('caption id: '+ str(self.query_labels[i].cpu().int().item()) + '\n')
                f.write(captions[i]+'\n')
            f.close()
            # print('only top10')
            top_index = index[:10]
            for ii, idx in enumerate(top_index):
                print('idx', idx, self.img_paths[idx])
                # print('self.img_paths[idx]', self.img_paths[idx])
                id =  self.gallery_labels[idx].cpu().int().item()
                name = 'top_'+str(ii)+'_'+'id_'+str(id)+'_'+self.img_paths[idx].split('/')[-1]
                im = cv2.imread(self.img_paths[idx])
                im = cv2.resize(im, (128, 384))
                font = cv2.FONT_HERSHEY_SIMPLEX
                im = cv2.putText(im, str(scores[idx]).zfill(4), (0, 20), font, 0.6, (199, 228, 250), 2)   
                # print('score[idx]', scores[idx])
                path = tmp_path+'/'+name
                cv2.imwrite(path, im)
                # return a
                # shutil.copy(self.img_paths[idx], tmp_path+'/'+name)
            # return a


       

