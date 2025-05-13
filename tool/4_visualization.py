import cv2
import os
# 将bbox可视化在视频上

# 输入文件路径
video_path = r"D:\DataSet\MOT-UAV\TestVideo\MultiUAV-299.mp4"  # 视频文件路径
gt_path = r"D:\DeepLearning_Projects\Tracking\DeepSORT_YOLOv5_Pytorch-master\output\predict\MultiUAV-299.txt"        # 标注文件路径
output_dir = r"D:\DataSet\MOT-UAV\Video"  # 输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取标注文件
with open(gt_path, "r") as f:
    lines = f.readlines()

# 解析标注信息
annotations = []
for line in lines:
    parts = line.strip().split(",")
    frame_id, obj_id, xmin, ymin, w, h = map(float, parts[:6])
    annotations.append((int(frame_id), int(obj_id), int(xmin), int(ymin), int(w), int(h)))

# 打开视频
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(output_dir, "video_result.mp4"), fourcc, fps, (frame_width, frame_height))

# 遍历每一帧
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # 获取当前帧的所有标注信息
    frame_annotations = [ann for ann in annotations if ann[0] == frame_id]

    for _, obj_id, xmin, ymin, w, h in frame_annotations:
        # 绘制边界框
        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (0,255,0), 1)
        # 绘制目标ID
        cv2.putText(frame, str(obj_id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # 写入帧到输出视频
    out.write(frame)

# 释放资源
cap.release()
out.release()
print("Processing complete. All objects have been saved to", output_dir)