from __future__ import division

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
# import matplotlib.pyplot as plt
from bbox import box_iou


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    Transform coord to model space
    @param prediction:
    @param inp_dim:
    @param anchors:
    @param num_classes:
    @param CUDA:
    @return: Tensor with same shape as argument prediction. Wherein, coord is transformed to cx,cy,h,w of model space.
    """
    # coord <-> input: bx = cx + sigmoid(tx)
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride  # How about directly getting the same from prediction.size(-1)?
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidence. Result is normalized into (0, 1)
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    # Equivalent: prediction[:, :, (0,1,4)] = torch.sigmoid(prediction[:, :, (0,1,4)])

    #Add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    # log space transform height and the width
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors  # Get positive h & w via exponential

    # Sigmoid the class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, :4] *= stride  # The multiplication by stride map bboxes to model space

    return prediction


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def get_im_dim(im):
    im = cv2.imread(im)
    w, h = im.shape[1], im.shape[0]
    return w, h


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def write_results(prediction, confidence, nms=True, nms_conf=0.4):
    """

    @param prediction: batch size * bboxes of all heads          * (bbox coord, confidence, classes，head)
                       n          * 10647                        * 86
           detail:     batch size * anchors of 3 heads * anchor# * (bbox coord, confidence, classes, head)
                       n          * (13*13+26*26+52*52) * 3      * (4         + 1         + 80     + 1   )
           bbox coord pattern: cx,cy,w,h,confidence,classes(x Nc),head
           Note: cx, cy, w and h in NN model space.
    @param confidence:
    @param nms:
    @param nms_conf:
    @return:
    """
    # Mask confidence matrix via filtering confidence in last dimension.
    # Call unsqueeze(2) to add dim-2 which was suppressed during filtering.
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    # conf_mask shape: Bsize x total of all bboxes x 1
    # equivalent: (prediction.select(2,4) > confidence).float().unsqueeze(-1)
    prediction = prediction * conf_mask  # Binarize prediction result

    # IMPROVEMENT Calculate num_classes from prediction instead of num_classes argument
    num_classes = prediction.size(2) - 6

    try:
        # Extract the indices of all bboxes with non-zero confidence. The bboxes are represented as a 2-D matrix,
        # shaped as B * bbox index. Then transpose the matrix to get dimension: bbox index * B
        ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except Exception as e:
        print('Exception in calling torch.nonzero(): %s' % repr(e), flush=True)
        return 0

    # Calculate diagonal ends coordinates of the boxes and place them in first 4 positions of last dim.
    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_a[:, :, :4]  # Overwrite bbox coord cx,cx,w,h -> x1,y1,x2,y2

    batch_size = prediction.size(0)

    output = prediction.new(1, prediction.size(2) + 1)  # Extend last dimension by 1 to keep image No.
    write = False

    for ind in range(batch_size):
        # Select the images from the batch without any classification info.
        # image_pred shape: total of all bboxes x 5 (bbox coord, confidence)
        image_pred = prediction[ind][:, :5]

        # Extract the classes having maximum score at a grid cell, and the indices of that classes.
        # Add the class scores and the class indices of the classes having maximum score to image_pred.
        max_cls_score, max_cls_index = torch.max(prediction[ind][:, 5:5 + num_classes], 1)
        max_cls_score = max_cls_score.float().unsqueeze(1)
        max_cls_index = max_cls_index.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_cls_score, max_cls_index, prediction[ind][:, -2:])
        image_pred = torch.cat(seq, 1)
        # image_pred shape: total of all bboxes x vector size
        # vector pattern: bbox coord, confidence, class score, class index, head, image id
        #              9:     4            1           1             1        1       1
        vectorsize = image_pred.size(-1)

        # Get rid of the zero-confidence entries
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))

        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, vectorsize).detach()
        # image_pred_ shape: total of all bboxes w/ non-zero confidences x vector size

        # Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_.detach()[:, -3])  # class index position -3
        except Exception as e:
            print('Exception in calling unique(): %s' % repr(e), flush=True)
            continue
        # WE will do NMS classwise
        for cls in img_classes:
            # get the detections with one particular class. Note: class index position is -3
            cls_mask = image_pred_ * (image_pred_[:, -3] == cls).float().unsqueeze(1)
            # cls_mask shape: (# of vectors with valid class scores) x (vector size)
            # Note for cla_mask: all zeros for the vectors with class <> cls
            class_mask_ind = torch.nonzero(cls_mask[:, 4]).squeeze()  # Slice objectiveness/confidence

            image_pred_class = image_pred_[class_mask_ind, :].view(-1, vectorsize)

            # sort the detections such that the entry with the maximum objectiveness/confidence
            #  is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]  # [1] indicates index part
            image_pred_class = image_pred_class[conf_sort_index]
            clscnt = image_pred_class.size(0)

            # if nms has to be done
            if nms:
                # For each detection
                for i in range(clscnt):
                    # Get the IOUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        # Calculate IoU between class i and class i+1 ~ class n
                        ious = box_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                    except ValueError:
                        break

                    except IndexError:
                        break

                    # Zero out all the detections that have IoU >= threshold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i + 1:] *= iou_mask

                    # Keep the non-zero entries following existing entries in image_pred_class in dim-0
                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, vectorsize)

            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to
            # We use a linear structure to hold ALL the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra batch column
            # image_pred_class shape: (# of eligible bboxes) x (vector size)
            # vector pattern: bbox coord, confidence, class score, class index, head, image id
            #              9:     4            1           1             1        1       1
            img_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = img_ind, image_pred_class
            # output entry pattern: img_id,x1,y1,x2,y2,confidence,class-prob,class-index
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
            # output shape: (# of eligible bboxes) x (new vector size)
            # vector pattern: image id, bbox coord, confidence, class score, class index, head, image id
            #             10:    1         4            1           1             1        1       1
    return output[..., :-1] if write else 0


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:12:16 2018

@author: ayooshmac
"""


def predict_transform_half(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    grid_size = inp_dim // stride

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    #Add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda().half()
        y_offset = y_offset.cuda().half()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    #log space transform height and the width
    anchors = torch.HalfTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    #Softmax the class scores
    prediction[:, :, 5: 5 + num_classes] = nn.Softmax(-1)(Variable(prediction[:, :, 5: 5 + num_classes])).data

    prediction[:, :, :4] *= stride

    return prediction


def write_results_half(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).half().unsqueeze(2)
    prediction = prediction * conf_mask

    try:
        ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_a[:, :, :4]

    batch_size = prediction.size(0)

    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]

        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.half().unsqueeze(1)
        max_conf_score = max_conf_score.half().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        #Get rid of the zero entries
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        if len(non_zero_ind.shape) == 0:
            continue
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, image_pred.size(1))
        except:
            continue

        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1].long()).half()

        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).half().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1, image_pred_.size(1))

            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            if len(image_pred_class.shape) == 1:
                image_pred_class = image_pred_class.unsqueeze(0)
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = box_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                    except ValueError:
                        break

                    except IndexError:
                        break

                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).half().unsqueeze(1)
                    image_pred_class[i + 1:] *= iou_mask

                    # Keep the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, image_pred_class.size(1))

            if len(non_zero_ind.size()) == 0:
                continue

            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1).to(output.device)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    return output
