from ultralytics import YOLO

import cv2

class YoloObjectDetection:
    mModel = None
    mDetectResult = []

    def __init__(self):
        #加载Yolo模型
        self.mModel = YOLO('../5 Realtime Dectect/yolov8n.pt', verbose=False)

    def getDetectResult(self):
        return self.mDetectResult

    # 调用Yolo模型进行目标检测
    def detectObjects(self, img):
        #调用Yolo模型进行预测并返回结果
        results = self.mModel(img, stream=True,verbose=False)

        self.mDetectResult = []

        # 读取检测的多个结果并保存
        for result in results:
            if len(result.boxes.cls) > 0:
                for i in range(len(result.boxes.cls)):
                    # 类别ID
                    leibie_id = int(result.boxes.cls[i].item())

                    # 类别
                    leibie = result.names[leibie_id]

                    # 相似度
                    xiangsidu = str(result.boxes.conf[i].item())[0:3]

                    # 坐标
                    zuobiao = result.boxes.xyxy[i].tolist()

                    self.mDetectResult.append({'类别': leibie, '相似度': xiangsidu, '坐标': zuobiao})

        return self.mDetectResult

    #把检测的结果在传入的图片上用方框标记出来
    def labelImg(self, img, detectResult = None):
        if detectResult is None:
            detectResult = self.mDetectResult

        if len(detectResult) > 0:
            for i in detectResult:
                # 画框
                cv2.rectangle(img, (int(i['坐标'][0]), int(i['坐标'][1])), (int(i['坐标'][2]), int(i['坐标'][3])),
                              (0, 255, 0), 2)

                # 标记类型
                cv2.putText(img, f" {i['类别']}", (int(i['坐标'][0]), int(i['坐标'][1]) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)

                # 相似度
                cv2.putText(img, f" {i['相似度']}", (int(i['坐标'][0]) + 80, int(i['坐标'][1]) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)



