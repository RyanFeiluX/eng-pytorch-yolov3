from __future__ import division

import argparse
import os
import os.path as osp
import shutil
# import json
import time
# import torch
# import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# import numpy as np
# import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, read_image, get_weight_config
import random
import pickle as pkl
# import itertools
from loss import LossCalculator
from torch.optim import lr_scheduler


def load_gtboxes(batch, scale=None, move=None):
    labels = []
    labeldir = osp.join(osp.split(batch[0])[0], 'labeled')
    for idx, img in enumerate(batch):
        for fn in os.listdir(labeldir):
            if not osp.isfile(osp.join(labeldir, fn)):
                continue
            if osp.splitext(fn) != (osp.splitext(osp.basename(img))[0], '.txt'):
                continue
            with open(osp.join(labeldir, fn), mode='r', encoding='utf-8') as fh:
                m = move[idx]
                s = scale[idx] if scale else 1.0
                for line in fh.readlines():
                    if len(line.strip()) == 0:
                        continue
                    line_ = [float(x) for x in line.strip().split(' ')]
                    cls_id = line_[0]
                    coords = line_[1:]
                    coords = np.array(coords) * s
                    coords[0:2] += m
                    labels.append([idx, cls_id] + coords.tolist())
    return labels  # pattern: img id, cls id, norm_x1, norm_y1, norm_x2, norm_y2


def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)


def arg_parse():
    """
    Parse arguements to the detect module
    
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images',
                        help="Directory containing images to perform training upon",
                        default="dataset", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions",
                        default=0.3)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--pt-model", dest='pretrained_model', help="pretrained_model",
                        default=None, type=str)
    parser.add_argument("--new-model", dest='new_model', help="New model to be generated or finetuned",
                        default="yolox", type=str)
    parser.add_argument("--cfg-ref", dest='cfg_ref', help="CFG file as model configuration",
                        default="yolov3", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. Increase to increase accuracy. "
                             "Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)

    args = arg_parse()

    if not osp.exists(args.cfg_ref):
        exit(-1)

    if args.new_model:
        fn_newcfg = osp.join(osp.dirname(args.cfg_ref), osp.splitext(osp.basename(args.new_model))[0] + '.cfg')
        if not osp.samefile(args.cfg_ref, fn_newcfg):
            shutil.copy(args.cfg_ref, fn_newcfg)
        # os.sync()
    else:
        fn_newcfg = args.cfg_ref

    scales = args.scales

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()
    device = 'cuda' if CUDA else 'cpu'

    # num_classes = 80
    classes = load_classes('data/coco.names')

    # Set up the neural network
    print("Loading network.....")
    weightsfile, _ = get_weight_config(args.pretrained_model)
    model = Darknet(fn_newcfg).train().to(device)
    if weightsfile:
        model.load_weights(weightsfile)
    else:
        init_weights(model)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # Set the model in training mode
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=float(model.net_info["learning_rate"]))  # adjust beta1 to momentum

    read_dir = time.time()
    # Training phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if
                  os.path.splitext(img)[1] == '.png'
                  or os.path.splitext(img)[1] == '.jpeg'
                  or os.path.splitext(img)[1] == '.jpg']
    except NotADirectoryError:
        imlist = [osp.join(osp.realpath('.'), images)]
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    if not os.path.exists('test'):
        os.makedirs('test')

    load_batch = time.time()

    batches = list(map(prep_image,
                       [read_image(imgfile) for imgfile in imlist],
                       [inp_dim for _ in range(len(imlist))],
                       [imgfile for imgfile in imlist]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
    im_path_list = [x[3] for x in batches]
    path_batches = [[x] for x in im_path_list]
    im_scale_list = [x[4] for x in batches]
    scale_batches = [[x] for x in im_scale_list]
    im_move_list = [x[5] for x in batches]
    move_batches = [[x] for x in im_move_list]

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 1 if len(im_dim_list) % batch_size else 0

    # if len(im_dim_list) % batch_size:
    #     leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                len(im_batches))])) for i in range(num_batches)]
        path_batches = [(im_path_list[i * batch_size: min((i + 1) * batch_size,
                                                          len(im_path_list))]) for i in range(num_batches)]
        scale_batches = [(im_scale_list[i * batch_size: min((i + 1) * batch_size,
                                                            len(im_scale_list))]) for i in range(num_batches)]
        move_batches = [(im_move_list[i * batch_size: min((i + 1) * batch_size,
                                                          len(im_move_list))]) for i in range(num_batches)]

    print('batch size: %d, batch total: %d' % (batch_size, len(im_batches)))

    anchors = model.net_info["anchors"].split(",")
    anchors = [(int(anchors[i]), int(anchors[i + 1])) for i in range(0, len(anchors), 2)]
    lcalc = LossCalculator(model.net_info, anchors, device)

    start_det_loop = time.time()

    epoches = int(model.net_info["epoches"])
    for epoch in range(epoches):
        bloss = 0.0
        output = None
        write = False
        for idx, batch in enumerate(im_batches):
            # print('Batch: %d' % idx, flush=True)

            pathbatch = path_batches[idx]
            scalebatch = scale_batches[idx]
            movebatch = move_batches[idx]

            # load the image
            start = time.time()
            if CUDA:
                batch = batch.cuda()

            # batch.requires_grad = True
            # batch.retain_grad()

            # Apply offsets to the result predictions
            # Transform the predictions as described in the YOLO paper
            # flatten the prediction vector
            # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
            # Put every proposed box as a row.
            # with torch.no_grad():

            prediction = model((batch), CUDA)
            # prediction: batch size * bboxes of all heads          * (bbox coord, confidence, classes, head)
            #             n          * 10647                         * 86
            # detail:     batch size * anchors of 3 heads * anchor# * (bbox coord, confidence, classes, head)
            #             n          * (13*13+26*26+52*52) * 3       * (4         + 1         + 80     + 1   )
            # example:    2          * 10647                         * 86


            # get the boxes with object confidence > threshold
            # Convert the coordinates to absolute coordinates
            # perform NMS on these boxes, and save the results
            # I could have done NMS and saving separately to have a better abstraction
            # But both these operations require looping, hence
            # clubbing these ops in one loop instead of two.
            # loops are slower than vectorised operations.

            # prediction = write_results(prediction, confidence, nms=True, nms_conf=nms_thresh)
            # # prediction: B * anchor boxes of all heads * (img id, bbox coord, confidence, class score, class id, head)
            # # detail:     B * anchor boxes of 3 heads   * (img id, bbox coord, confidence, class score, class id, head)
            # #             n * (13*13+26*26+52*52) * 3   * (1     + 4         + 1         + 1          + 1       + 1   )
            # # example:    2 * 10647                     * 9

            end = time.time()

            # Per write_results() implementation, return integer 0 in case of nothing detected.
            if type(prediction) is int:
                for im_num, image in enumerate(imlist[idx * batch_size: min((idx + 1) * batch_size, len(imlist))]):
                    im_id = idx * batch_size + im_num
                    objs = []
                    print("{0:20s} predicted in {1:6.3f} seconds".format(os.path.basename(image),
                                                                         (end - start) / batch_size))
                    print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
                    print("----------------------------------------------------------")
                if output is None:
                    output = prediction
                continue

            # prediction[:, 0] = prediction[:, 0] + idx * batch_size

            # if not write:
            #     output = prediction
            #     write = True
            # else:
            #     assert type(output) not in [int, None], "Internal error: output is unavailable so far."
            #     assert output.shape[1] == prediction.shape[1], (
            #         "Mismatch sizes in dim=1 during concatenate two tensors!!! {} vs {}".format(output.shape,
            #                                                                                     prediction.shape))
            #     output = torch.cat((output, prediction))

            # for im_num, image in enumerate(imlist[idx * batch_size: min((idx + 1) * batch_size, len(imlist))]):
            #     im_id = idx * batch_size + im_num
            #     objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            #     print("{0:20s} predicted in {1:6.3f} seconds".format(os.path.basename(image),
            #                                                          (end - start) / batch_size))
            #     print("{0:20s} {1:s}".format("Objects Detected:", ", ".join(objs)))
            #     print("----------------------------------------------------------", flush=True)

            gtboxes = load_gtboxes(pathbatch,
                                   scale=scalebatch,
                                   move=movebatch)
            loss, loss_items = lcalc.calc_loss(prediction, gtboxes)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if CUDA:
                torch.cuda.synchronize()

            bloss += loss
        print('Epoch: %3d\tLoss: %.4f' % (epoch, bloss))

    # if output is None:
    #     print("No detections were made")
    #     exit()
    # elif type(output) is int:
    #     print("No object was detected")
    #     exit()

    # im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    #
    # scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)
    #
    # output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    # output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
    #
    # output[:, 1:5] /= scaling_factor
    #
    # for idx in range(output.shape[0]):
    #     # Forced type cast for green code inspection
    #     output[idx, [1, 3]] = torch.clamp(output[idx, [1, 3]],
    #                                       torch.Tensor([0.0]).to('cuda' if CUDA else 'cpu'), im_dim_list[idx, 0])
    #     output[idx, [2, 4]] = torch.clamp(output[idx, [2, 4]],
    #                                       torch.Tensor([0.0]).to('cuda' if CUDA else 'cpu'), im_dim_list[idx, 1])
    #
    output_recast = time.time()
    #
    # class_load = time.time()
    #
    # colors = pkl.load(open("pallete", "rb"))
    #
    # draw = time.time()


    def write(x, batches, results):
        c1 = tuple(x[1:3].int()) if not CUDA else tuple(x[1:3].cpu().int().numpy())
        c2 = tuple(x[3:5].int()) if not CUDA else tuple(x[3:5].cpu().int().numpy())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, thickness=2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        t_margin = (4, 6)
        c2 = c1[0] + t_size[0] + t_margin[0] * 2, c1[1] + t_size[1] + t_margin[1] * 2
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0]+t_margin[0], c1[1] + t_size[1] + t_margin[1]),
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img

    # list(map(lambda x: write(x, im_batches, orig_ims), output))
    #
    # det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, os.path.basename(x)))
    #
    # list(map(cv2.imwrite, det_names, orig_ims))  # det_names and orig_ims as arguments of cv2.imwrite()

    end = time.time()

    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Learning (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
    # print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    # print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
    print("----------------------------------------------------------")

    # Save weights
    fn_weights = osp.join('models', args.new_model + '.weights')
    model.save_weights(fn_weights)

    torch.cuda.empty_cache()
