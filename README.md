# Darknet YOLOv4, YOLOv3 to ONNX
## Description

## Installation
- torch 1.9.1
- numpy
- opencv-python
- onnx 1.7.0
- onnxruntime-gpu 1.7.0
- onnx-simplifier

## Usage
```bash
# darknet cfg & weights
python3 demo.py --cfg cfg/yolov4.cfg --weight weights/yolov4.weights --img data/dog.jpg
python3 demo.py --cfg cfg/yolov4-csp.cfg --weight weights/yolov4-csp.weights --img data/dog.jpg
python3 demo.py --cfg cfg/yolov3.cfg --weight weights/yolov3.weights --img data/dog.jpg

# custom cfg & weights
# batch-size -1 means dynamic ONNX model
python3 demo.py --cfg path/to/your/cfg --weight path/to/your/weights --img path/to/your/img --batch-size -1 --score 0.8 --nms 0.3 --out mymodel.onnx
```

## YOLO Spec.
### `mask` in `[yolo]`
https://github.com/pjreddie/darknet/issues/558#issuecomment-376041045

### Decode Parts in Different Version of YOLO Model
https://github.com/WongKinYiu/ScaledYOLOv4/issues/202#issuecomment-810913378
