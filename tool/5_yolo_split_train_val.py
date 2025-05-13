import os
import shutil
import random

# split YOLO_val from YOLO_train

# 定义源目录和目标目录
train_images_dir = r"/mnt/sda/Disk_C/MultiUAV/Yolo_train/images"
train_labels_dir = r"/mnt/sda/Disk_C/MultiUAV/Yolo_train/labels"
val_images_dir = r"/mnt/sda/Disk_C/MultiUAV/Yolo_val/images"
val_labels_dir = r"/mnt/sda/Disk_C/MultiUAV/Yolo_val/labels"

# 创建目标目录（如果不存在）
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 获取所有图片文件名
image_files = [f for f in os.listdir(train_images_dir) if f.endswith(".jpg")]
random.shuffle(image_files)  # 随机打乱文件顺序

# 计算拆分点
split_index = int(len(image_files) * 0.8)

# 分割训练集和验证集
train_images = image_files[:split_index]
val_images = image_files[split_index:]

# 复制验证集的图片和标注文件到目标目录
for image_file in val_images:
    # 复制图片
    src_image_path = os.path.join(train_images_dir, image_file)
    dst_image_path = os.path.join(val_images_dir, image_file)
    shutil.move(src_image_path, dst_image_path)  # 使用 shutil.move 移动文件

    # 复制对应的标注文件
    label_file = image_file.replace(".jpg", ".txt")
    src_label_path = os.path.join(train_labels_dir, label_file)
    dst_label_path = os.path.join(val_labels_dir, label_file)
    shutil.move(src_label_path, dst_label_path)  # 使用 shutil.move 移动文件

print(f"Total images: {len(image_files)}")
print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print("Dataset split completed successfully.")