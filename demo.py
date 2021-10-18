import time
import argparse

import numpy as np
import cv2
import onnxruntime

from darknet2onnx import transform_to_onnx


WARMUP_STEPS = 30
INFERENCE_STEPS = 30


"""
nms & multiclass_nms_class_aware are borrowed from YOLOX
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py
"""


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def get_detections(predictions, score_thresh=0.5, nms_thresh=0.45):
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    dets = multiclass_nms_class_aware(boxes, scores, nms_thr=nms_thresh, score_thr=score_thresh)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        return final_boxes, final_scores, final_cls_inds
    else:
        return None, None, None


def plot_boxes_cv2(image_path, boxes, scores, cls_inds, savename=None, class_names=None):
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]
    for b, s, c in zip(boxes, scores, cls_inds):
        x1 = int(b[0] * width)
        y1 = int(b[1] * height)
        x2 = int(b[2] * width)
        y2 = int(b[3] * height)
        c = int(c)
        t = class_names[c] if class_names else str(c)
        t += ": {:.2f}".format(s)
        img = cv2.putText(img, t, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def detect(session, image_path, score_thresh=0.1, nms_thresh=0.45):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]
    t1 = time.time()
    image_src = cv2.imread(image_path)
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)  # HWC to CHW
    img_in /= 255.0
    img_in = np.expand_dims(img_in, axis=0)
    t2 = time.time()

    input_name = session.get_inputs()[0].name
    # warm-up
    for _ in range(WARMUP_STEPS):
        _ = session.run(None, {input_name: img_in})  # output = [[batch, num, 4 + num_classes]]

    # inference
    t3 = time.time()
    for _ in range(INFERENCE_STEPS):
        outputs = session.run(None, {input_name: img_in})  # output = [[batch, num, 4 + num_classes]]
    t4 = time.time()
    final_boxes, final_scores, final_cls_inds = get_detections(outputs[0][0], score_thresh, nms_thresh)
    t5 = time.time()

    # analysis
    print("Preprocessing : {:.4f}s".format(t2 - t1))
    print("Inference     : {:.4f}s".format((t4 - t3) / INFERENCE_STEPS))
    print("Postprocessing: {:.4f}s".format(t5 - t4))
    print("Total         : {:.4f}s".format(t2 - t1 + (t4 - t3) / INFERENCE_STEPS + t5 - t4))
    return final_boxes, final_scores, final_cls_inds


def main(args):
    onnx_path_demo = transform_to_onnx(args.cfg, args.weight, args.out, args.batch_size)
    session = onnxruntime.InferenceSession(onnx_path_demo)
    final_boxes, final_scores, final_cls_inds = detect(session, args.img, args.score, args.nms)
    plot_boxes_cv2(args.img, final_boxes, final_scores, final_cls_inds, savename='predictions_onnx.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Darknet to ONNX")
    parser.add_argument("--cfg", "-c", type=str)
    parser.add_argument("--weight", "-w", type=str)
    parser.add_argument("--batch-size", "-b", default=1, type=int, help="If batch size > 0, ONNX model will be static. If batch size <= 0, ONNX model will be dynamic")
    parser.add_argument("--score", default=0.5, type=float)
    parser.add_argument("--nms", default=0.45, type=float)
    parser.add_argument("--img", "-i", type=str)
    parser.add_argument("--out", "-o", default="model.onnx")
    args = parser.parse_args()
    for k, v in vars(args).items():
        print("{}: {}".format(k, v))
    main(args)
