# MM-AntiUAV: A Comprehensive Benchmark for Multi-UAV Tracking and Intent Recognition

This repository contains sample evaluation and training code for multi drone tracking and intent recognition. To address this task, we held The [4th Anti UAV Workshop at CVPR25](https://anti-uav.github.io)

## Prepare 
1 Create a virtual environment with Python >=3.8  
~~~
conda create -n py38 python=3.8    
conda activate py38   
~~~

2 Install pytorch >= 1.6.0, torchvision >= 0.7.0.
~~~
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
~~~


3 Install all dependencies
~~~
pip install -r requirements.txt
~~~
## Dataset
raw video Dataset and labels:
~~~
Dataset/
   ├── TrainVideos/
   │       ├── MultiUAV-001.mp4
   │       └── ....
   │
   └── TrainLabels/
           ├── MultiUAV-001.txt
           └── ....
~~~
The annotation format for each video is as follows:

    frame ID, object ID, x1, y1, w, h, confidence=1, class=1, visibility ratio=1.0]

## Train YOLOV5
1.First, you need to convert the training dataset into the following format to train YOLOv5:
~~~
path/to/dataset/
    ├── Yolo_train/
    │   ├── images/
    │   │   ├── 000001.jpg
    │   │   ├── 000002.jpg
    │   │   └── ....
    │   └── labels/
    │       ├── 000001.txt
    │       ├── 000002.txt
    │       └── ....
    └── Yolo_val/
        ├── images/
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   └── ....
        └── labels/
            ├── 000001.txt
            ├── 000002.txt
            └── ....
~~~
Annotation information for each frame is stored in a TXT file, with the format shown below:
~~~
1     0.057547       0.177881      0.043969      0.047285
1     0.070156       0.332100      0.039094      0.041230
1     0.114109       0.394297      0.036344      0.039922
1     0.146406       0.475557      0.036406      0.036816
......
~~~
    which means:`Class_ID, X Center, Y Center, Width, Height`


2.Modify the YAML file `models/yolov5s-MultiUAV.yaml` for yolov5.

3.Modify the YAML file `data/MOT-UAV.yaml` for Dataset.

4.run:
~~~
python yolov5/train.py --batch 16 --epochs 30 --cfg models/yolov5s-MultiUAV.yaml --data data/MOT-UAV.yaml --device 0 --single-cls
~~~
now you can get the `.pt` weight and put it into `yolov5/weights/`.

**In the Baseline provided by this code, we trained Yolov5 for 30 epochs.**

## Train DeepSort 
Train DeepSORT in any way you prefer.

And place `ckpt.t7` file under `deep_sort/deep/checkpoint/`

## Test
Modify the path configuration in `test_videos.py` and then run it:
~~~
python test_videos.py
~~~

We also provide a Python script `4_visualization.py` to visualize the bounding boxes on videos.

## Submit
The submission format is provided in `Submission_example.zip`, where each line represents an object. Each value corresponds to 

**[frame ID, object ID, x1, y1, w, h, confidence, class, visibility ratio]**.

(The first six values are the most important: **frame ID, object ID, x1, y1, w, h**, where x1 and y1 represent the coordinates of the top-left corner.)



## Reference
1) [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)   
2) [yolov5](https://github.com/ultralytics/yolov5)  
3) [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)       
4) [deep_sort](https://github.com/nwojke/deep_sort)   

Note: please follow the [LICENCE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) of YOLOv5! 
