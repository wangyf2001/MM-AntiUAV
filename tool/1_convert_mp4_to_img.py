import cv2
import os

# convert video -->> Yolo format , 把原始视频和标签转换成YOLO格式，以便于训练

# 定义路径 images and labels will save here:
output_frames_folder = r'/mnt/sda/Disk_C/MultiUAV/Yolo_train/images'  # 输出帧的文件夹路径
output_labels_folder = r'/mnt/sda/Disk_C/MultiUAV/Yolo_train/labels'  # 输出标注文件的文件夹路径

# 创建输出文件夹
os.makedirs(output_frames_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

global_frame_id = 0

# 读取标注文件
def read_annotations(annotation_path):
    annotations = {}
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            xmin, ymin, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            class_id = int(parts[6])

            # 转换为YOLO格式：class_id, x_center, y_center, width, height, 方便给Yolo训练
            if xmin + w > 640:   #默认图片大小：640*512
                w = 640 - xmin
            if ymin + h > 512:
                h = 512 - ymin

            x_center = (xmin + w / 2) / 640  # 假设视频宽度为640
            y_center = (ymin + h / 2) / 512  # 假设视频高度为512
            width = w / 640
            height = h / 512

            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            if frame_id not in annotations:
                annotations[frame_id] = []
            annotations[frame_id].append((class_id, x_center, y_center, width, height))
    return annotations


def process_video_and_label(video_path, label_path):
    # 获取视频文件路径
    video_path = video_path
    annotation_path = label_path
    annotations = read_annotations(annotation_path)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()
    global global_frame_id
    local_frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        global_frame_id += 1
        local_frame_id += 1
        frame_filename = f"{global_frame_id:06d}.jpg"
        label_filename = f"{global_frame_id:06d}.txt"

        # 保存帧为图片
        frame_path = os.path.join(output_frames_folder, frame_filename)
        cv2.imwrite(frame_path, frame)

        # 保存标注文件
        label_path = os.path.join(output_labels_folder, label_filename)
        with open(label_path, 'w') as f:
            if local_frame_id in annotations:
                for annotation in annotations[local_frame_id]:
                    class_id, x_center, y_center, width, height = annotation
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            else:
                # 如果当前帧没有标注，标注文件为空
                pass

    print("Processing complete.")
    cap.release()

if __name__ == '__main__':
    # 视频目录和标签目录
    video_dir = r"/mnt/sda/Disk_C/MultiUAV/TrainVideo/"
    label_dir = r"/mnt/sda/Disk_C/MultiUAV/TrainLabel/"

    # 遍历视频目录中的所有视频文件
    for video_filename in os.listdir(video_dir):
        if video_filename.endswith(".mp4"):  # 确保是视频文件
            video_path = os.path.join(video_dir, video_filename)

            # 构造对应的标签路径
            video_name = video_filename.replace("MultiUAV-", "UAV").replace(".mp4", "") #UAV001
            label_path = os.path.join(label_dir, video_name, ".txt")

            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                print(f"Label file not found for video: {video_filename}")
                continue

            print("Start:",video_path," ",label_path)
            # 调用处理函数
            process_video_and_label(video_path, label_path)