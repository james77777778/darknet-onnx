import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def yolo_forward_dynamic(output, num_classes, anchors, num_anchors, scale_x_y, version="yolov4"):
    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)

        bxy_list.append(output[:, begin: begin + 2])
        bwh_list.append(output[:, begin + 2: begin + 4])
        det_confs_list.append(output[:, begin + 4: begin + 5])
        cls_confs_list.append(output[:, begin + 5: end])

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
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(output.size(0), num_anchors * output.size(2) * output.size(3), num_classes)

    # Adopt different decoding method based on
    # https://github.com/WongKinYiu/ScaledYOLOv4/issues/202#issuecomment-810913378
    if version == "yolov4" or version == "yolov3":
        bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
        bwh = torch.exp(bwh)
        det_confs = torch.sigmoid(det_confs)
        cls_confs = torch.sigmoid(cls_confs)
    elif version == "scaled-yolov4":
        bxy = bxy * scale_x_y - 0.5 * (scale_x_y - 1)
        bwh = torch.pow(bwh * 2, 2)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(3) - 1, output.size(3)), axis=0).repeat(output.size(2), 0), axis=0), axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(2) - 1, output.size(2)), axis=1).repeat(output.size(3), 1), axis=0), axis=0)

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
        bx = bxy[:, ii: ii + 1] + torch.tensor(grid_x, device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1: ii + 2] + torch.tensor(grid_y, device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii: ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1: ii + 2] * anchor_h[i]

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
    """nn.Upsample is deprecated"""

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class YOLOLayer(nn.Module):
    ''' [yolo] in darknet cfg'''
    def __init__(self, anchor_mask=[], anchors=[], num_classes=80, num_anchors=9, stride=32, scale_x_y=1.0, new_coords=1):
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
            self.version = "yolov4"
        elif self.new_coords == 1:
            self.version = "scaled-yolov4"

    def forward(self, x):
        # training
        if self.training:
            return x

        # inference
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]
        return yolo_forward_dynamic(x, self.num_classes, masked_anchors, len(self.anchor_mask), self.scale_x_y, self.version)
