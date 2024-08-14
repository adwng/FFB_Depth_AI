# FFB Grading

| Folder | Description  |
|--|--|
| `Depth Detection` | Contains script to interface with Intel Realsense using pyrealsense and collect color and depth frames |
| `FFB_Camera` | Includes a simple training script file|

## Depth Detection

### Dependencies - Win10/11
1. Intel RealSense Viewer
2. `pyrealsense`

> [!TIP]
> `DatasetPreparation.py` and `DatasetPrepration_v2.py` has included basic scripts for interfacing with the RealSense camera and export the images in different picture formats

<details>
  
<summary>TODO</summary>

- Find Annotation Tool for RGBD Data
- Find relevant AI architecture for training RGBD data, can take a look at extended Mask-RCNN

</details>


## FFB Camera 

>[!IMPORTANT]
>The remainder of the code can be found inside the RPI5, the script in the RPI5 can be used as an executable.
>For more information on training with YOLOv8 or logic implementation for counting, please check [here](https://github.com/adwng/OPENCV-YOLOv8}).
