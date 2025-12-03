from ultralytics import YOLO

def 训练Yolov8模型():
    """
    训练Yolov8模型
    """
        # 直接加载模型（不要先 yaml 再 load，会创建两份模型！）
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        device=0,
        batch=4,          # 关！键！ batch=16 8GB 显存一定爆
        workers=0,        # Windows 下必须改为 0，否则显存会被复制多份
        cache=False,      # cache=True 会把 dataset 全放显存，非常容易 OOM
        close_mosaic=10,
    )

if __name__ == '__main__':
    训练Yolov8模型()