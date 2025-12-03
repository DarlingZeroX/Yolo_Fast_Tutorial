import 数据集划分
import 数据集复制
import 构建DataYaml
from ultralytics import YOLO

if __name__ == '__main__':
    file_path = r"datasets\images"
    txt_path = r'datasets\labels'
    new_file_path = r"imageset"

    # 这里是正常划分数据集，按比例划分
    # 这里就是 训练集占 8，验证集占 1, 测试集占 1
    数据集划分.划分数据集(
        file_path,
        txt_path,
        new_file_path,
        train_rate=0.8,
        val_rate=0.1,
        test_rate=0.1
    )
    """
    # 这里在数据不够的使用，训练集验证集测试集都使用全部的图片
    # 可能会过拟合，识别成功率会增加，但准确率会降低
    数据集复制.数据集复制(
        file_path,
        txt_path,
        new_file_path,
    )
    """

    # 这里构建模型训练所需要的data.yaml文件
    构建DataYaml.构建DataYaml(
        imageset_path = new_file_path, classes_path ='datasets/labels/classes.txt'
    )

    # 直接加载模型
    model = YOLO('yolo11n.pt')

    # 训练Yolov8模型
    # 模型训练完成后会控制台提示训练好的模型在哪，注意看，一般在run文件下的【train+数字】目录下
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        device=0,
        batch=4,
        workers=0,
        cache=False,
        close_mosaic=10,
        # 保存更多信息，便于后续使用
        save=True,
        save_period=-1,
        project='runs/detect',  # 明确指定输出目录
        name='train',
        exist_ok=True
    )

    # 训练完成后自动复制best.pt到当前目录
    import shutil
    import os

    # 获取最新的训练目录
    train_dir = 'runs/detect/train'
    if os.path.exists(train_dir):
        best_weight = os.path.join(train_dir, 'weights', 'best.pt')
        if os.path.exists(best_weight):
            shutil.copy(best_weight, './best.pt')
            print(f"✅ 已复制最佳权重到当前目录: {os.path.abspath('./best.pt')}")
        else:
            print("❌ 未找到best.pt文件")

