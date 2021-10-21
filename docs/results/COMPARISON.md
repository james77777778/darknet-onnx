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
|YOLOv4|Darknet|19.275ms|1.1GB|
|YOLOv4|ONNX|20.852ms|0.67GB|
|YOLOv3|Darknet|16.932ms|0.98GB|
|YOLOv3|ONNX|18.324ms|0.72GB|
