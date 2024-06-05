from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, read_image, get_weight_config
import pandas as pd
import random
import pickle as pkl
import itertools


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers = num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5, 5) for x in range(num_layers)])
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_


def arg_parse():
    """
    Parse arguements to the detect module
    
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images',
                        help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det',
                        help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions",
                        default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--pt_model", dest='pretrained_model', help="pretrained_model",
                        default="yolov3", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. Increase to increase accuracy. "
                             "Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()

    scales = args.scales

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes('data/coco.names')

    # Set up the neural network
    print("Loading network.....")
    weightsfile, cfgfile = get_weight_config(args.pretrained_model)
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU available, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    read_dir = time.time()
    # Detection phase
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

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()

    batches = list(map(prep_image,
                       [read_image(imgfile) for imgfile in imlist],
                       [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 1 if len(im_dim_list) % batch_size else 0

    # if len(im_dim_list) % batch_size:
    #     leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                len(im_batches))])) for i in range(num_batches)]

    print('batch size: %d, batch total: %d' % (batch_size, len(im_batches)))

    write = False
    # model(get_test_input(inp_dim, CUDA), CUDA)

    start_det_loop = time.time()

    output = None
    for idx, batch in enumerate(im_batches):
        print('Batch: %d' % idx, flush=True)

        # load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()

        # Apply offsets to the result predictions
        # Transform the predictions as described in the YOLO paper
        # flatten the prediction vector
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        #        prediction = prediction[:,scale_indices]

        # get the boxes with object confidence > threshold
        # Convert the coordinates to absolute coordinates
        # perform NMS on these boxes, and save the results
        # I could have done NMS and saving separately to have a better abstraction
        # But both these operations require looping, hence
        # clubbing these ops in one loop instead of two.
        # loops are slower than vectorised operations.

        prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)

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

        prediction[:, 0] += idx * batch_size

        if not write:
            output = prediction
            write = True
        else:
            assert type(output) not in [int, None], "Internal error: output is unavailable so far."
            assert output.shape[1] == prediction.shape[1], (
                "Mismatch sizes in dim=1 during concatenate two tensors!!! {} vs {}".format(output.shape,
                                                                                            prediction.shape))
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[idx * batch_size: min((idx + 1) * batch_size, len(imlist))]):
            im_id = idx * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(os.path.basename(image),
                                                                 (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ", ".join(objs)))
            print("----------------------------------------------------------", flush=True)

        if CUDA:
            torch.cuda.synchronize()

    if output is None:
        print("No detections were made")
        exit()
    elif type(output) is int:
        print("No object was detected")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for idx in range(output.shape[0]):
        # Forced type cast for green code inspection
        output[idx, [1, 3]] = torch.clamp(output[idx, [1, 3]],
                                          torch.Tensor([0.0]).to('cuda' if CUDA else 'cpu'), im_dim_list[idx, 0])
        output[idx, [2, 4]] = torch.clamp(output[idx, [2, 4]],
                                          torch.Tensor([0.0]).to('cuda' if CUDA else 'cpu'), im_dim_list[idx, 1])

    output_recast = time.time()

    class_load = time.time()

    colors = pkl.load(open("pallete", "rb"))

    draw = time.time()


    def write(x, batches, results):
        c1 = tuple(x[1:3].int()) if not CUDA else tuple(x[1:3].cpu().int().numpy())
        c2 = tuple(x[3:5].int()) if not CUDA else tuple(x[3:5].cpu().int().numpy())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, thickness=2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img


    list(map(lambda x: write(x, im_batches, orig_ims), output))

    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, os.path.basename(x)))

    list(map(cv2.imwrite, det_names, orig_ims))  # det_names and orig_ims as arguments of cv2.imwrite()

    end = time.time()

    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()
