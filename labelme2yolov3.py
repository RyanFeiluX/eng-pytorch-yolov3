import argparse
import os
import os.path as osp
import json


def parse_labelme(labeldir):
    labels = []
    for fn in os.listdir(labeldir):
        gtboxes = []
        if not osp.isfile(osp.join(labeldir, fn)):
            continue
        if osp.splitext(fn)[1] != '.json':
            continue
        with open(osp.join(labeldir, fn), mode='r', encoding='utf-8') as fh:
            dict_label = json.load(fh)
            shapes = dict_label['shapes']
            for shape in shapes:
                if shape['shape_type'] != 'rectangle':
                    continue
                point0 = [shape['points'][0][0], shape['points'][0][1]]
                point1 = [shape['points'][1][0], shape['points'][1][1]]
                label = shape['label']
                if label in labels:
                    labelid = labels.index(label)
                else:
                    labelid = len(labels)
                    labels.append(label)
                line = [str(labelid)] + ['{}'.format(p) for p in point0 + point1]
                gtboxes.append(line)
        if len(gtboxes) > 0:
            with open((osp.splitext(osp.join(labeldir, fn))[0] + '.txt'), mode='w+t', encoding='utf-8') as fh:
                [fh.write(' '.join(line) + '\n') for line in gtboxes]
                fh.flush()
    namesdir = osp.abspath(osp.join(labeldir, '..', 'names/yolox.names'))
    if not osp.exists(osp.split(namesdir)[0]):
        os.mkdir(osp.split(namesdir)[0])
    with open(namesdir, mode='w+t', encoding='utf-8') as fh:
        fh.write('\n'.join(labels))
        fh.write('\n')
        fh.flush()


def arg_parse():
    parser = argparse.ArgumentParser(description='Convert LableMe format to yolov3 format')

    parser.add_argument("--labels-dir", dest='labels_dir',
                        help="Directory containing LabelMe output",
                        default="dataset/labels", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    parse_labelme(args.labels_dir)
