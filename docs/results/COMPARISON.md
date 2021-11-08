# Darknet vs. DarknetONNX
## Visualization
|Model|Framework|Image|
|-|-|-|
|YOLOv4|Darknet|<img src="darknet_yolov4_predictions.jpg" alt="drawing" width="75%"/>|  
|YOLOv4|ONNX|<img src="onnx_yolov4_predictions.jpg" alt="drawing" width="75%"/>|
|YOLOv3|Darknet|<img src="darknet_yolov3_predictions.jpg" alt="drawing" width="75%"/>|  
|YOLOv3|ONNX|<img src="onnx_yolov3_predictions.jpg" alt="drawing" width="75%"/>|
|YOLOv4-csp|Darknet|<img src="darknet_yolov4-csp_predictions.jpg" alt="drawing" width="75%"/>|  
|YOLOv4-csp|ONNX|<img src="onnx_yolov4-csp_predictions.jpg" alt="drawing" width="75%"/>|
|YOLOv4-tiny|Darknet|<img src="darknet_yolov4-tiny_predictions.jpg" alt="drawing" width="75%"/>|  
|YOLOv4-tiny|ONNX|<img src="onnx_yolov4-tiny_predictions.jpg" alt="drawing" width="75%"/>|

## Inference Speed & GPU Mem.
Tested on Ubuntu 18.04 with 1080ti & i7-7800X

|Model|Framework|Time per 1 Image|GPU Mem.|
|-|-|-|-|
|YOLOv4|Darknet|19.7ms|1.13GB|
|YOLOv4|ONNX|20.9ms|1.46GB|
|YOLOv3|Darknet|17.0ms|1.03GB|
|YOLOv3|ONNX|16.9ms|1.40GB|
