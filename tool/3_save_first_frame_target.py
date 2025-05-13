import os
# 只保留第一帧目标

# 定义文件夹路径
input_folder = r"D:\DataSet\MOT-UAV\Output_TestLabels"  # 输入文件夹路径
output_folder = r"D:\DataSet\MOT-UAV\Output_FirstFrameOnly"  # 输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有文件
files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

# 遍历每个文件
for file in files:
    input_file_path = os.path.join(input_folder, file)
    output_file_path = os.path.join(output_folder, file)

    # 初始化一个列表来存储第一帧的数据
    first_frame_data = []

    # 读取文件并处理每一行
    with open(input_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 0:  # 确保行不为空
                frame_id = parts[0]  # 获取帧号
                if frame_id == '1':  # 只保留第一帧的数据
                    first_frame_data.append(line)

    # 将第一帧的数据写入输出文件
    with open(output_file_path, 'w') as f:
        for line in first_frame_data:
            f.write(line)

print("处理完成，第一帧的目标已保存到输出文件夹。")