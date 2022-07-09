# Darknet YOLOv4, YOLOv3 to ONNX
## To Do
- [ ] Create Python package
- [x] (22/03/03) Support FP16 (half precision) conversion (`--to_float16`)
- [x] (21/11/09) Support YOLOv3, YOLOv4, YOLOv4-csp, YOLOv4-tiny conversion

## Description
This project can convert original AlexeyAB/darknet model weights & cfg to ONNX format.  

`main.py` shows all the steps as following:  
1. Export darknet weights to ONNX format via PyTorch
2. Run the inference including preprocessing & postprocessing
3. Visualize the result

Supported models:  
- [x] YOLOv4
- [x] YOLOv3
- [x] YOLOv4-csp (Scaled-YOLOv4)
- [x] YOLOv4-tiny

Other models are not tested but you can try

### Inference Result
Use `data/dog.jpg` as the example:

YOLOv4
- Darknet
    ```bash
    bicycle: 92%    (left_x:  114   top_y:  128     width:  458     height:  299)
    dog: 98%        (left_x:  129   top_y:  225     width:  184     height:  317)
    truck: 92%      (left_x:  464   top_y:   77     width:  221     height:   93)
    pottedplant: 33%        (left_x:  681   top_y:  109     width:   37     height:  45)
    ```
    <img src="docs/results/darknet_yolov4_predictions.jpg" alt="drawing" width="75%"/>
- DarknetONNX
    ```bash
    bicycle: 92%  (left_x:  114   top_y:  127     width:  458     height:  299)
    dog: 98%      (left_x:  128   top_y:  224     width:  185     height:  317)
    truck: 92%    (left_x:  463   top_y:  76      width:  221     height:  93)
    pottedplant: 33%        (left_x:  681   top_y:  109     width:  36      height:  45)
    ```
    <img src="docs/results/onnx_yolov4_predictions.jpg" alt="drawing" width="75%"/>

More visualizations & Inference speed comparison can be found at [docs/results/COMPARISON.md](docs/results/COMPARISON.md).  

## Installation
- torch >= 1.9.1 (>= 1.9.1 for `torch.nn.Mish` activation)
- opencv-python
- onnxruntime 1.9.0
- onnxmltools 1.10.0
- packaging

```bash
pip install -r requirements.txt
```

## Usage
1. Prepare pretrained model weights or your custom model weights  
    ```bash
    mkdir weights
    wget -O weights/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
    wget -O weights/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
    ```
2. Convert Darknet model
    - Convert and visualize the result:
        ```bash
        # help
        python3 main.py -h
        
        # darknet cfg & weights
        python3 main.py --cfg cfg/yolov3.cfg --weight weights/yolov3.weights --img data/dog.jpg --names data/coco.names
        python3 main.py --cfg cfg/yolov4.cfg --weight weights/yolov4.weights --img data/dog.jpg --names data/coco.names
        python3 main.py --cfg cfg/yolov4-csp.cfg --weight weights/yolov4-csp.weights --img data/dog.jpg --names data/coco.names
        
        # custom cfg & weights
        python3 main.py --cfg cfg/yolov4-obj.cfg --weight weights/yolov4-obj.weights --img your.jpg --names your.names
        
        # it will show index if not specifying `--names`
        python3 main.py --cfg cfg/yolov4.cfg --weight weights/yolov4.weights --img data/dog.jpg
        ```
        Outputs `model.onnx` and `onnx_predictions.jpg`.
    - Only convert the model (use standalone `darknetonnx.darknet`)
        ```bash
        # help
        python3 -m darknetonnx.darknet -h

        # darknet yolov3
        python3 -m darknetonnx.darknet --cfg cfg/yolov3.cfg --weight weights/yolov3.weights

        # darknet yolov3 with float16
        python3 -m darknetonnx.darknet --cfg cfg/yolov3.cfg --weight weights/yolov3.weights --to-float16
        ```

## YOLO Spec.
### `mask` in `[yolo]`
https://github.com/pjreddie/darknet/issues/558#issuecomment-376041045

### Decode Parts in Different Version of YOLO Models
YOLO Layer:
https://github.com/WongKinYiu/ScaledYOLOv4/issues/202#issuecomment-810913378

## Credit
- https://github.com/AlexeyAB/darknet
- https://github.com/Tianxiaomo/pytorch-YOLOv4
- https://github.com/Megvii-BaseDetection/YOLOX

## Q&A
### `TypeError: export() got an unexpected keyword argument 'example_outputs'`
`torch.onnx._export` has deprecated the keyword argument `example_outputs` with `torch > 1.10.1`.

The newest version of this repository has fixed the issue.
