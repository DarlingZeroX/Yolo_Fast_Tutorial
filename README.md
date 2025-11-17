# YOLO快速入门教程

[YOLOv8快速入门视频](https://www.bilibili.com/video/BV1o142147Gi/?spm_id_from=0.0.upload.video_card.click)
本教程将指导您从环境配置到实际应用，快速掌握使用YOLOv8进行目标检测的基本流程。

## 目录结构

```
Yolo_Fast_Tutorial/
├── 1 Python Enviroment/    # Python环境配置
├── 2 Cuda/                  # CUDA配置
├── 3 Yolo Enviroment/       # YOLO环境配置
├── 4 Detect Picture/        # 图像目标检测示例
├── 5 Realtime Dectect/      # 实时目标检测示例
├── LICENSE
└── README.md                # 项目说明文档
```

## 教程步骤

### 1. Python环境配置

首先需要安装Python和PyCharm开发环境：

- **Python官网**：[www.python.org](http://www.python.org)
- **PyCharm官网**：[https://www.jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download)

### 2. CUDA配置

为了加速深度学习计算，我们需要配置CUDA环境：

- **PyTorch官网**：[https://pytorch.org](https://pytorch.org)
- **CUDA安装地址**：[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

请根据您的显卡型号选择合适的CUDA版本。

### 3. YOLO环境配置

在命令行中执行以下命令安装YOLO相关依赖：

```bash
# 安装ultralytics包（YOLOv8的官方库）
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

# 卸载可能存在的不兼容版本
pip uninstall torch
pip uninstall torchvision

# 安装对应CUDA版本的PyTorch（示例为CUDA 12.1）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

使用`TorchSupport.py`检查PyTorch环境是否正确配置：

```python
import torch

def checkTorch():
    print("torch版本", torch.__version__)
    print("GPU是否可用", torch.cuda.is_available())
    print("GPU个数", torch.cuda.device_count())
    print("对应cudnn版本号", torch.backends.cudnn.version())
    print("对应cuda版本号", torch.version.cuda)
    print("--------------------------------------------------")
    # 查看torchvision和torch版本是否匹配
    import torchvision
    print(torch.__version__)
    print(torchvision.__version__)

if __name__ == "__main__":
    checkTorch()
```

### 4. 图像目标检测

在`4 Detect Picture`目录中，我们提供了一个简单的图像目标检测示例：

- **YoloObjectDetection.py**：YOLO目标检测的核心实现类
- **Main.py**：主程序，演示如何使用YOLO进行图像检测

使用方法：

1. 将需要检测的图片命名为`test.png`，放在该目录下
2. 运行`Main.py`
3. 检测结果将保存在`save.png`中

### 5. 实时目标检测

在`5 Realtime Dectect`目录中，我们提供了一个实时屏幕截图目标检测示例：

- **YoloObjectDetection.py**：YOLO目标检测的核心实现类
- **Main.py**：主程序，演示如何进行实时目标检测

使用方法：

1. 运行`Main.py`
2. 程序会实时截取屏幕左上角区域（可在代码中修改`rect`变量调整截取区域）
3. 按`Q`键退出程序

## 核心代码说明

### YoloObjectDetection类

```python
class YoloObjectDetection:
    mModel = None
    mDetectResult = []

    def __init__(self):
        # 加载YOLOv8模型
        self.mModel = YOLO('yolov8n.pt', verbose=False)

    # 进行目标检测
    def detectObjects(self, img):
        # 调用YOLO模型进行预测
        results = self.mModel(img, stream=True, verbose=False)
        # 处理检测结果...

    # 在图像上标记检测结果
    def labelImg(self, img, detectResult=None):
        # 在图像上绘制边界框和标签...
```

## 注意事项

1. 确保您的电脑有NVIDIA显卡并正确安装了CUDA
2. 首次运行时，YOLOv8模型会自动下载
3. 可以修改代码中的参数，如检测阈值、显示方式等
4. 实时检测可能会占用较高的系统资源

## 参考资源

- [Ultralytics YOLOv8官方文档](https://docs.ultralytics.com/)
- [PyTorch官方文档](https://pytorch.org/docs/stable/)
- [OpenCV官方文档](https://docs.opencv.org/)

## 许可证

本项目采用MIT许可证。详见LICENSE文件。
