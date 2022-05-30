import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxmltools.utils import load_model, save_model
from onnxmltools.utils.float16_converter import convert_float_to_float16


'''
Export Function
'''


def export_to_onnx(cfgfile, weightfile, outputfile, batch_size=1, to_float16=False):
    # load model
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    model.eval()
    model.fuse()

    # prepare ONNX config
    input_names = ['input']
    output_names = ['output']
    dynamic = False
    dynamic_axes = None
    if batch_size < 0:
        dynamic = True
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    b = -1 if dynamic else batch_size
    if b < 0:
        b = 1
    x = torch.randn((b, 3, model.height, model.width))

    # export the model
    traced_model = torch.jit.trace(model, x, check_trace=False)
    f = 'model.pt'
    torch.jit.save(traced_model, f)
    loaded_model = torch.jit.load(f)
    torch.onnx._export(
        loaded_model,
        x,
        outputfile,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        example_outputs=loaded_model(x),
    )
    os.remove(f)

    # export to float16 model
    if to_float16:
        onnx_model = load_model(outputfile)
        half_onnx_model = convert_float_to_float16(onnx_model)
        save_model(half_onnx_model, outputfile)

    return outputfile


'''
Modules & Darknet Model
'''


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))
    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


def yolo_forward_dynamic(output, num_classes, anchors, num_anchors, scale_x_y, version='yolov4'):
    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)

        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3))

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(output.size(0), num_anchors, num_classes, output.size(2) * output.size(3))
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes]
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(
        output.size(0), num_anchors * output.size(2) * output.size(3), num_classes
    )

    # Adopt different decoding method based on
    # https://github.com/WongKinYiu/ScaledYOLOv4/issues/202#issuecomment-810913378
    if version == 'yolov4' or version == 'yolov3':
        bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1.0)
        bwh = torch.exp(bwh)
        det_confs = torch.sigmoid(det_confs)
        cls_confs = torch.sigmoid(cls_confs)
    elif version == 'scaled-yolov4':
        bxy = bxy * scale_x_y - 0.5 * (scale_x_y - 1)
        bwh = torch.pow(bwh * 2, 2)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(
        np.expand_dims(
            np.expand_dims(np.linspace(0, output.size(3) - 1, output.size(3)), axis=0).repeat(output.size(2), 0), axis=0
        ),
        axis=0,
    )
    grid_y = np.expand_dims(
        np.expand_dims(
            np.expand_dims(np.linspace(0, output.size(2) - 1, output.size(2)), axis=1).repeat(output.size(3), 1), axis=0
        ),
        axis=0,
    )

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii : ii + 1] + torch.FloatTensor(grid_x).to(device)
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1 : ii + 2] + torch.FloatTensor(grid_y).to(device)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii : ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)

    #######################################
    #   Figure out bboxes from slices     #
    #######################################

    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = torch.cat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= output.size(3)
    by_bh /= output.size(2)

    # Shape: [batch, num_anchors * H * W]
    bx = bx_bw[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3))
    by = by_bh[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3))
    bw = bx_bw[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3))
    bh = by_bh[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3))

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx - bw * 0.5 + bw
    by2 = by - bh * 0.5 + bh

    # Shape: [batch, num_anchors * h * w, 4]
    boxes = torch.stack((bx1, by1, bx2, by2), dim=-1)
    # boxes:     [batch, num_anchors * H * W, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W, 1]
    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    confs = cls_confs * det_confs
    # confs:       [batch, num_anchors * H * W, num_classes]
    # predictions: [batch, num_anchors * H * W, 4 + num_classes]
    predictions = torch.cat([boxes, confs], dim=-1)
    return predictions


class Upsample(nn.Module):
    '''nn.Upsample is deprecated'''

    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class YOLOLayer(nn.Module):
    '''[yolo] in darknet cfg'''

    def __init__(
        self, anchor_mask=[], anchors=[], num_classes=80, num_anchors=9, stride=32, scale_x_y=1.0, new_coords=1
    ):
        super(YOLOLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.stride = stride
        self.scale_x_y = scale_x_y
        self.new_coords = new_coords
        # parse new_coords
        if self.new_coords == 0:
            self.version = 'yolov4'
        elif self.new_coords == 1:
            self.version = 'scaled-yolov4'

    def forward(self, x):
        # training
        if self.training:
            return x

        # inference
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step : (m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]
        return yolo_forward_dynamic(
            x, self.num_classes, masked_anchors, len(self.anchor_mask), self.scale_x_y, self.version
        )


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.training = False
        self.blocks = self.parse_cfg(cfgfile)
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])
        self.models = self.create_network(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.train(self.training)  # initialize to eval mode

    def parse_cfg(self, cfgfile):
        blocks = []
        fp = open(cfgfile, 'r')
        block = None
        line = fp.readline()
        while line != '':
            line = line.rstrip()
            if line == '' or line[0] == '#':
                line = fp.readline()
                continue
            elif line[0] == '[':
                if block:
                    blocks.append(block)
                block = dict()
                block['type'] = line.lstrip('[').rstrip(']')
                # set default value
                if block['type'] == 'convolutional':
                    block['batch_normalize'] = 0
                if block['type'] == 'yolo':
                    block['new_coords'] = 0
                    block['scale_x_y'] = 1.0
            else:
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                block[key] = value
            line = fp.readline()
        if block:
            blocks.append(block)
        fp.close()
        return blocks

    def forward(self, x):
        ind = -2
        outputs = dict()
        final_outputs = []
        for block in self.blocks:
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'upsample']:
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        x = outputs[layers[0]]
                        outputs[ind] = x
                    else:
                        groups = int(block['groups'])
                        group_id = int(block['group_id'])
                        _, b, _, _ = outputs[layers[0]].shape
                        x = outputs[layers[0]][:, b // groups * group_id : b // groups * (group_id + 1)]
                        outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
                elif len(layers) == 4:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x3 = outputs[layers[2]]
                    x4 = outputs[layers[3]]
                    x = torch.cat((x1, x2, x3, x4), 1)
                    outputs[ind] = x
                else:
                    print('rounte number > 2 ,is {}'.format(len(layers)))

            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'yolo':
                out = self.models[ind](x)
                final_outputs.append(out)
            else:
                print('unknown type %s' % (block['type']))

        if self.training:
            return final_outputs
        else:
            return torch.cat(final_outputs, dim=1)  # concate all yolo layer outputs

    def create_network(self, blocks):
        models = nn.ModuleList()
        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module(
                        'conv{0}'.format(conv_id),
                        nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False),
                    )
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module(
                        'conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad)
                    )
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                elif activation == 'mish':
                    model.add_module('mish{0}'.format(conv_id), nn.Mish(inplace=True))
                elif activation == 'logistic':
                    model.add_module('logistic{0}'.format(conv_id), nn.Sigmoid())
                elif activation == 'linear':
                    model.add_module('linear{0}'.format(conv_id), nn.Identity())
                else:
                    print('convolution has no activation: {}'.format(activation))
                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                model = nn.Sequential()
                if pool_size == 2 and stride == 1:
                    model.add_module('zeropad{0}'.format(conv_id), nn.ZeroPad2d((0, 1, 0, 1)))
                model.add_module(
                    'maxpool{0}'.format(conv_id),
                    nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=(pool_size - 1) // 2),
                )
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)
                models.append(Upsample(scale_factor=stride, mode='nearest'))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        prev_filters = out_filters[layers[0]]
                        prev_stride = out_strides[layers[0]]
                    else:
                        prev_filters = out_filters[layers[0]] // int(block['groups'])
                        prev_stride = out_strides[layers[0]] // int(block['groups'])
                elif len(layers) > 1:
                    assert layers[0] == ind - 1 or layers[1] == ind - 1
                    prev_filters = sum([out_filters[i] for i in layers])
                    prev_stride = out_strides[layers[0]]
                else:
                    print('route error!!!')
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(nn.Identity())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(nn.Identity())
            elif block['type'] == 'yolo':
                anchors = block['anchors'].split(',')
                anchors = [float(i) for i in anchors]
                anchor_mask = block['mask'].split(',')
                anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer = YOLOLayer(
                    anchor_mask,
                    anchors,
                    int(block['classes']),
                    int(block['num']),
                    prev_stride,
                    float(block['scale_x_y']),
                    int(block['new_coords']),
                )
                self.num_classes = int(block['classes'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                print('unknown type %s' % (block['type']))
        return models

    def load_conv(self, buf, start, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(torch.from_numpy(buf[start : start + num_b]))
        start = start + num_b
        conv_model.weight.data.copy_(torch.from_numpy(buf[start : start + num_w]).reshape(conv_model.weight.data.shape))
        start = start + num_w
        return start

    def load_conv_bn(self, buf, start, conv_model, bn_model):
        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()
        bn_model.bias.data.copy_(torch.from_numpy(buf[start : start + num_b]))
        start = start + num_b
        bn_model.weight.data.copy_(torch.from_numpy(buf[start : start + num_b]))
        start = start + num_b
        bn_model.running_mean.copy_(torch.from_numpy(buf[start : start + num_b]))
        start = start + num_b
        bn_model.running_var.copy_(torch.from_numpy(buf[start : start + num_b]))
        start = start + num_b
        conv_model.weight.data.copy_(torch.from_numpy(buf[start : start + num_w]).reshape(conv_model.weight.data.shape))
        start = start + num_w
        return start

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = self.load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = self.load_conv(buf, start, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'yolo':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        # print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1 :])
                        break
            fused_list.append(a)
        self.module_list = fused_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Darknet to ONNX')
    parser.add_argument('--cfg', '-c', type=str, required=True, help='Darknet .cfg file')
    parser.add_argument('--weight', '-w', type=str, required=True, help='Darknet .weights file')
    parser.add_argument(
        '--batch-size',
        '-b',
        default=1,
        type=int,
        help='If batch size > 0, ONNX model will be static. If batch size < 0, ONNX model will be dynamic. Skip 0',
    )
    parser.add_argument('--out', '-o', default='model.onnx', help='Output file path')
    parser.add_argument('--to-float16', action='store_true', help='Use onnxmltools to convert to float16 model')
    args = parser.parse_args()
    export_to_onnx(args.cfg, args.weight, args.out, args.batch_size, args.to_float16)
