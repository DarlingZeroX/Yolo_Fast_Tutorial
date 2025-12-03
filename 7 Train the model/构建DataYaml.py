import os

def 构建DataYaml(imageset_path = 'imageset',classes_path='datasets/labels/classes.txt'):
    """
    构建Yolov8训练所需的data.yaml文件
    
    返回:
        str: 生成的data.yaml文件路径
    """
    # 项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 读取classes.txt文件
    # classes_file = os.path.join(project_root, 'datasets', 'labels', 'classes.txt')
    classes_file = os.path.join(project_root, classes_path)
    print(classes_file)
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    # 检查imageset目录结构
    imageset_dir = os.path.join(project_root, imageset_path)
    train_images_dir = os.path.join(imageset_dir, 'train', 'images')
    val_images_dir = os.path.join(imageset_dir, 'val', 'images')
    test_images_dir = os.path.join(imageset_dir, 'test', 'images')
    
    # 如果imageset目录不存在，输出警告
    if not os.path.exists(imageset_dir):
        print(f"警告: {imageset_dir} 目录不存在，数据集可能尚未划分")
        print("请先运行'数据集划分脚本.py'来划分数据集")
    
    # 创建data.yaml内容
    data_yaml_content = [
        '# Yolov8 Dataset Configuration',
        '',
        '# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2]',
        'train: imageset/train/images',  # 相对路径
        'val: imageset/val/images',      # 相对路径
        'test: imageset/test/images',    # 相对路径
        '',
        '# Number of classes',
        f'nc: {len(classes)}',
        '',
        '# Class names',
        'names: [' + ', '.join([f"'{cls}'" for cls in classes]) + ']'
    ]
    
    # 保存data.yaml文件
    data_yaml_path = os.path.join(project_root, 'data.yaml')
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data_yaml_content) + '\n')
    
    # 输出信息
    print(f"data.yaml文件已创建: {data_yaml_path}")
    print(f"类别数量: {len(classes)}")
    print(f"类别名称: {', '.join(classes)}")
    print("数据集路径配置:")
    print(f"  训练集: imageset/train/images")
    print(f"  验证集: imageset/val/images")
    print(f"  测试集: imageset/test/images")
    
    return data_yaml_path


if __name__ == '__main__':
    # 执行构建data.yaml的函数
    构建DataYaml()
