import cv2
import mss
import numpy as np

from YoloObjectDetection import YoloObjectDetection

#实例化Yolo目标检测类
yod = YoloObjectDetection()

#实时截屏的区域
rect = {"left" :0, "top": 0, "width": 1000, "height": 500}

with mss.mss() as m:
    while True:
        #截屏
        img = m.grab(rect)
        #转成numpy数组
        img = np.array(img)
        #这里要转出Yolo能输入的格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #进行目标检测
        yod.detectObjects(img)

        #在图片上用方框标记出来
        yod.labelImg(img)

        #实时显示图片
        cv2.imshow("DNF Yolo Auto", img)

        #按Q键退出
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break