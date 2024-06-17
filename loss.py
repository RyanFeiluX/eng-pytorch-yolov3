# encoding: utf-8
from __future__ import division

import json
import os, sys
import torch
import torch.nn as nn
from bbox import box_iou
import numpy as np


def SmoothBCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class LossCalculator:
    def __init__(self, hyp, anchors, device):
        self.device = device
        self.hyp = hyp
        self.inp_dim = int(hyp['height'])
        self.num_class = int(hyp['classes'])
        self.num_abox = 3
        self.num_head = 3  # 3 scales of feature map
        list_ha = json.loads(hyp["head_anchor"])
        self.gridsizes = [ha['grids'] for ha in list_ha]
        anchors_ = self.hyp['anchors'].split(',')
        anchors = [(int(anchors_[i].strip()), int(anchors_[i + 1].strip())) for i in range(0, len(anchors_), 2)]
        a = [[] for _ in range(len(list_ha))]
        for ha in list_ha:
            a[ha['head']] = [np.array(anchors[m]) * ha['grids'] / self.inp_dim for m in ha['mask']]
            # a[ha['head']] = [np.array(anchors[m]) for m in ha['mask']]
        self.anchors = torch.from_numpy(np.array(a)).to(self.device)
        hyp['truth_thresh'].sort(key=lambda x: x['head'])
        self.gr = [tt['truth_thresh'] for tt in hyp['truth_thresh']]

        # Define criteria for classification loss and confidence loss
        self.balance = [.9, .9, .9]
        self.autobalance = False
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(self.hyp['cls_pw'])], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(self.hyp['obj_pw'])], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3. cp/cn for positive/negative label.
        self.cp, self.cn = SmoothBCE(eps=self.hyp.get('label_smoothing', 0.0))  # positive, negative BCE targets

    def build_targets(self, predictions, targets):
        """
        Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        @param predictions: list of feature map, dtype=tensor
           item shape: Bsize x anchor#/point of current scale x grid size x grid size x bbox attr
           example:    2     x 3                              x 13        x 13          85
                       2 x 3 x 13 x 13 x 85
        @param targets: tensor
           shape:   GT box# of current batch x (img id of batch, class id, coord(cx,cy,w,h))
           example: 5                        x (1              + 1       + 4               )
                    5 x 6
        @return: class, box, indices, and anchors.
        """

        # Na: Num of prediction heads, namely num of scales of final feature maps. 3 for YOLO v3.
        # Nt: Num of ground truth objects. It is from training dataset.
        na, nt = self.num_abox, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # gain is used to map targets([na,nt,7]) from normalized xywh to corresponding feature map scale
        # 7: Composed of image_index + class + xywh + anchor_index
        #                     1          1      4         1
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        # aidx for late use，bbox - anchor box mapping. Shape: Na x Nt
        aidx = (torch.arange(na, device=self.device)
                .float().view(na, 1).repeat(1, nt))  # same as .repeat_interleave(nt)
        # Repeat targets up to the num of anchors of current feature map. As a result, each bbox changes to 3 times.
        # It reaches 1-1 mapping b/w GT box entries and anchors. Try to compute IoU of each pair of GT box entry &
        # anchors for final high-confidence pairs. It is possible there are >1 anchors mapped to a single GT box.
        # targets: N x 6 -repeat-> Na x Nt x 6 -concat-> Na x Nt x (6 + 1)
        targets = torch.cat((targets.repeat(na, 1, 1), aidx[..., None]), 2)  # append anchor indices

        # 这两个变量是用来扩展正样本的 因为预测框预测到target有可能不止当前的格子预测到了
        # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
        # 设置网络中心偏移量
        g = 0.5  # bias 用来衡量target中心点离哪个格子近
        # off: Define a set grid offset to locate current grid cell and its 4 neighbor grid cells
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm  4 neighbor grid cells with same corner points
            ],
            device=self.device).float() * g  # offsets

        # Processing per feature map
        for i in range(self.num_head):  # 三个尺度的预测特征图输出分支
            # 当前feature map对应的三个anchor尺寸
            # anchors and prediction of current feature map
            anchors, shape = self.anchors[i], predictions[i].shape
            # [1, 1, 1, 1, 1, 1, 1] -> [1, 1, gw, gh, gw, gh, 1] pattern=image_index+class+xywh+anchor_index
            # gw/gh : Current feature map width/height in grid
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain  todo yxhw pattern in prediction?

            # Match targets to anchors
            targets[..., 2:6] /= self.inp_dim  # Scale xywh to scope [0,1]
            t = targets * gain  # shape(3,nt,7). Map targets' xywh to current feature map
            if nt:  # Matching only if any GT box available.
                # Matches
                # t: Na x Nt x 7 -> Na x Nt x 2
                # anchors: Na x 2 -> Na x 1 x 2
                # r: Na x Nt x 2
                r = t[..., 4:6] / anchors[:, None]  # w/h ratios:  gt_w/an_w, gt_h/an_h

                # Determine positive/negative sample of current GT via comparing width & height of GT boxes and anchors.

                # Pick out max of width and height ratios of GT boxes vs anchors per grid cell, including gt_w/an_w,
                # gt_h/an_h, an_w/gt_w and an_h/gt_h. If less than pre-defined threshold, the grid cell is taken as
                # a positive sample of the GT. Otherwise, negative.
                # 1st .max: Pick out max one of each pair of item from both Na x Nt x 2 tensors and achieve another
                # Na x Nt x 2 tensor.
                # 2nd .max: Pick max one in dim-2. Return max values and their indices with same shape Na x Nt
                # j: bool. Na x Nt  True: Current anchor is positive sample of current GT box. False: Negative sample.
                j = torch.max(r, 1 / r).max(2)[0] < float(self.hyp['anchor_t'])  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # t: Na x Nt x 7  j: Na x Nt -> t: Ntrue x 7  Ntrue = true# in j
                t = t[j]  # filter out highly-deviated anchor-GT pairs

                # Offsets 筛选当前格子周围格子 找到2个离target中心最近的两个格子  可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
                # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 也就是说一个目标需要3个格子去预测(计算损失)
                # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个 用这三个格子去预测这个目标(计算损失)
                # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                if gxy.shape[0] > 0:
                    assert gxi.view(-1).min() >= 0, "A grid xy falls out of the canvas."
                # 筛选中心坐标 距离当前grid_cell的左、上方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # j: [126] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
                # k: [126] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                # 筛选中心坐标 距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # l: [126] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
                # m: [126] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                # j: [5, 126]  torch.ones_like(j): 当前格子, 不需要筛选全是True  j, k, l, m: 左上右下格子的筛选结果
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 得到筛选后所有格子的正样本 格子数<=3*126 都不在边上等号成立
                # t: [126, 7] -> 复制5份target[5, 126, 7]  分别对应当前格子和左上右下格子5个格子
                # j: [5, 126] + t: [5, 126, 7] => t: [378, 7] 理论上是小于等于3倍的126 当且仅当没有边界的格子等号成立
                t = t.repeat((5, 1, 1))[j]
                # 添加偏移量
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()  # 预测真实框的网格所在的左上角坐标(有左上右下的网格)
            gi, gj = gij.T  # grid indices

            # Append
            # gj: 网格的左上角y坐标  gi: 网格的左上角x坐标
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def __calc__(self, bs, p, tcls, tbox, indices, anch):
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lcls = torch.zeros(1, device=self.device)

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.num_class), 1)  # subset of predictions

                # Box regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anch[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = box_iou(pbox, tbox[i], grad_needed=True, CIoU=True).squeeze()  # iou(prediction, target)
                lbox = lbox + (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                # if self.sort_obj_iou:
                j = iou.argsort()
                b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr[i] < 1:
                    iou = (1.0 - self.gr[i]) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.num_class > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls = lcls + self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj = lobj + obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[0] for x in self.balance]
        lbox = lbox * float(self.hyp["box"])
        lobj = lobj * float(self.hyp["obj"])
        lcls = lcls * float(self.hyp["cls"])

        loss = (lbox + lobj + lcls) * bs
        return loss, torch.cat((lbox, lobj, lcls)).detach()

    def calc_loss(self, predictions, gtboxes):
        # predictions: B * total anchors * (img id, bbox coord, confidence, class score, class id, head)
        # example:     2 x 10647         x (1     + 4         + 1         + 1          + 1       + 1   )
        # detail: total anchors = anchor box # per anchor point * sum of anchor points of each head
        # example:        10647 = 3                             x (13 x 13 + 26 x 26 + 52 x 52)
        # gtboxes: list of GT per image. n x 6
        #   item as list: img id in the batch, class id, cx,    cy,    w,     h
        #   example:           0                 3       0.1    0.2    1.5    0.6

        # *** Organize argument p ***
        vectorsize = predictions.size(-1)
        num_anchors = len(self.anchors[0])
        num_head = self.num_head
        p = []
        bs = predictions.size(0)
        for img in range(bs):
            prediction = predictions[img].view(-1, vectorsize)
            # first, phead = True, None
            for head in range(num_head):
                mask = (prediction[:, -1] == head).float().unsqueeze(-1)
                p_ = prediction[:, 5].clone() * mask
                ind_nz = torch.nonzero(p_[:, 5]).view(-1)
                if len(ind_nz.shape) == 0:
                    continue
                if len(ind_nz) not in [507, 2028, 8112]:
                    print("Internal error. Illegal tensor size: %d" % len(ind_nz))
                b0 = (prediction[ind_nz, :-1])
                b1 = b0.view(1, num_anchors, self.gridsizes[head], self.gridsizes[head], vectorsize - 1)
                if head < len(p):
                    p[head] = torch.cat((p[head], b1), 0)
                else:
                    p.append(b1)
                # for ind in ind_nz:
                #     b0 = torch.tensor(prediction[ind, :])
                #     b1 = b0.repeat(self.gridsizes[head], 1).repeat(self.gridsizes[head], 1, 1).repeat(num_anchors, 1, 1, 1).unsqueeze(0)
                #     # b0 = torch.zeros([1, self.gridsizes[head], self.gridsizes[head], 6])
                #     # b0[..., 2:6] = prediction[ind, 2:6]
                #     # b1 = b0.repeat(1, num_anchors, 1, 1, 1)
                #     if first:
                #         phead = b1
                #         first = False
                #     else:
                #         phead = torch.cat((phead, b1), 1)
            # if phead:
            #     p.append(phead)
            assert len(p) == num_head, 'All heads should have detection. ' + p
        # p: list of prediction per feature map, dtype=tensor
        #   item shape: Bsize x anchor#/point of current scale x grid size x grid size x bbox attr
        #   example:    2     x 3                              x 13        x 13          85

        # *** Organize argument targets ***
        targets = torch.tensor(gtboxes, device=self.device).view(-1, len(gtboxes[0]))
        # Transform targets from x1,y1,x2,y2 to cx,cxy,w,h
        cx, cy = (targets[:, 2] + targets[:, 4]) / 2, (targets[:, 3] + targets[:, 5]) / 2
        w, h = abs(targets[:, 4] - targets[:, 2]), abs(targets[:, 5] - targets[:, 3])
        targets[..., 2:6] = torch.stack((cx, cy, w, h), -1)
        # targets: tensor
        #   shape:   GT box# of current batch x (img id of batch, class id, coord(cx,cy,w,h))
        #   example: 34                       x (1              + 1       + 4               )

        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        return self.__calc__(bs, p, tcls, tbox, indices, anchors)
