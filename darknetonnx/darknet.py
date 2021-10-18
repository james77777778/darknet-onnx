import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Upsample, YOLOLayer


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.training = True
        self.blocks = self.parse_cfg(cfgfile)
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])
        self.models = self.create_network(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.train(self.training)  # initialize to training mode

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
                        x = outputs[layers[0]][:, b // groups * group_id:b // groups * (group_id + 1)]
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
                    print("rounte number > 2 ,is {}".format(len(layers)))

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
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
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
                    print("convolution has no activation: {}".format(activation))
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
                    model.add_module(
                        'zeropad{0}'.format(conv_id),
                        nn.ZeroPad2d((0, 1, 0, 1))
                    )
                model.add_module(
                    'maxpool{0}'.format(conv_id),
                    nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=(pool_size - 1) // 2)
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
                models.append(Upsample(scale_factor=stride, mode="nearest"))
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
                    assert(layers[0] == ind - 1 or layers[1] == ind - 1)
                    prev_filters = sum([out_filters[i] for i in layers])
                    prev_stride = out_strides[layers[0]]
                else:
                    print("route error!!!")
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
                    int(block['new_coords'])
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
        conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
        conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape))
        start = start + num_w
        return start

    def load_conv_bn(self, buf, start, conv_model, bn_model):
        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()
        bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
        bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
        bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
        bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
        conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape))
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
