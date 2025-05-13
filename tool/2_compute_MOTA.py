import os
import motmetrics as mm

# compute MOTA

mota_total = 0.0

def compute_mota(gt_file, result_file):
    # 读取文件
    gt = mm.io.loadtxt(gt_file, fmt="mot16", min_confidence=1)  # 读取真值文件，仅保留置信度为1的行
    ts = mm.io.loadtxt(result_file, fmt="mot16")  # 读取跟踪结果文件
    # 创建accumulator并计算MOTA
    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)  # 使用IOU作为距离度量，阈值为0.5
    # 计算指标
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'num_false_positives', 'num_misses', 'num_switches'], name='acc')

    mota_value = summary['mota'].iloc[0]  # 获取MOTA值
    fp_value = summary['num_false_positives'].iloc[0]
    fn_value = summary['num_misses'].iloc[0]
    sw_value = summary['num_switches'].iloc[0]

    return mota_value , fp_value , fn_value, sw_value

def process_directory(directory1, directory2, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    # 获取两个目录中的所有文件
    files1 = [f for f in os.listdir(directory1) if f.endswith(".txt")]

    i=0
    num_seq = len(files1)
    # 遍历每个文件
    for file in files1:
        gt_file = os.path.join(directory1, file)
        result_file = os.path.join(directory2, file)
        if not os.path.exists(result_file):
            continue
        output_file = os.path.join(output_folder, "mota_output.txt")

        mota_value , fp_value , fn_value, sw_value = compute_mota(gt_file, result_file)
        global mota_total
        mota_total += mota_value
        # 将MOTA值写入输出文件
        with open(output_file, 'a') as f:
            f.write(f"{file}:\t MOTA: {mota_value:.6f},\t FP: {fp_value}, \t FN:{fn_value},\t SW:{sw_value}"+'\n')

        i += 1
        print(f"[{i}/{num_seq}] \t MOTA: {mota_value:.6f},\t FP: {fp_value}, \t FN:{fn_value},\t SW:{sw_value}")


    mota_total = mota_total/num_seq
    print("All Completed, motal_total",mota_total)

if __name__ == '__main__':
    directory1 = r'D:\DataSet\MOT-UAV\Output_TestLabels_All' #GT
    directory2 = r'D:\DeepLearning_Projects\Tracking\DeepSORT_YOLOv5_Pytorch-master\output\predict' #predict
    output_folder = r'D:\DeepLearning_Projects\Tracking\DeepSORT_YOLOv5_Pytorch-master\output\moat' #save

    process_directory(directory1, directory2, output_folder)