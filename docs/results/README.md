# Darknet vs. DarknetONNX
## Visualization
|Model|Framework|Image|
|-|-|-|
|YOLOv4|darknet|<img src="darknet_yolov4_predictions.jpg" alt="drawing" width="75%"/>|  
|YOLOv4|darknetonnx|<img src="onnx_yolov4_predictions.jpg" alt="drawing" width="75%"/>|
|YOLOv3|darknet|<img src="darknet_yolov3_predictions.jpg" alt="drawing" width="75%"/>|  
|YOLOv3|darknetonnx|<img src="onnx_yolov3_predictions.jpg" alt="drawing" width="75%"/>|
|YOLOv4-csp|darknet|<img src="darknet_yolov4-csp_predictions.jpg" alt="drawing" width="75%"/>|  
|YOLOv4-csp|darknetonnx|<img src="onnx_yolov4-csp_predictions.jpg" alt="drawing" width="75%"/>|
|YOLOv4-tiny|darknet|<img src="darknet_yolov4-tiny_predictions.jpg" alt="drawing" width="75%"/>|  
|YOLOv4-tiny|darknetonnx|<img src="onnx_yolov4-tiny_predictions.jpg" alt="drawing" width="75%"/>|

## Inference Speed & GPU Mem.
Tested on Ubuntu 18.04 with 1080ti & i7-7800X

|Model|Framework|Time per 1 Image|GPU Mem.|
|-|-|-|-|
|YOLOv4|Darknet|19.275ms|1.1GB|
|YOLOv4|DarknetONNX|20.852ms|0.67GB|
|YOLOv3|Darknet|16.932ms|0.98GB|
|YOLOv3|DarknetONNX|18.324ms|0.72GB|
