import cv2
import numpy as np

"""
_COLORS & vis are borrowed from YOLOX
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/visualize.py
"""


_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def vis(
    image_path,
    boxes,
    scores,
    cls_ids,
    conf=0.5,
    class_names=None,
    out_img=None,
    print_bbox=False,
):
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]
    # zipped[0][0]: x0 of box
    sorted_bcs = sorted(
        zip(boxes, cls_ids, scores), key=lambda zipped: zipped[0][0]
    )
    for box, cls_id, score in sorted_bcs:
        cls_id = int(cls_id)
        if class_names is None:
            label = cls_id
        else:
            label = class_names[cls_id]
        if score < conf:
            continue
        x0 = int(box[0] * width)
        y0 = int(box[1] * height)
        x1 = int(box[2] * width)
        y1 = int(box[3] * height)
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = "{}: {:.2f}".format(label, score)
        txt_color = (
            (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.6, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 - int(1.5 * txt_size[1])),
            (x0 + txt_size[0] + 1, y0 - 1),
            txt_bk_color,
            -1,
        )
        cv2.putText(
            img,
            text,
            (x0, y0 - int(0.5 * txt_size[1])),
            font,
            0.6,
            txt_color,
            thickness=1,
        )
        if print_bbox:
            print(
                f"{label}: {score * 100.0:.0f}%\t(left_x:  {x0}\ttop_y:  {y0}\t"
                f"width:  {x1 - x0}\theight:  {y1 - y0})"
            )
    if out_img:
        print("save visulization to {}".format(out_img))
        cv2.imwrite(out_img, img)
    return img
