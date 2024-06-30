import cv2
from YoloObjectDetection import YoloObjectDetection

#实例化Yolo目标检测类
yod = YoloObjectDetection()

#读取需要检测的图片
jt = cv2.imread("test.png")

#进行目标检测
yod.detectObjects(jt)

#在图片上用方框标记出来
yod.labelImg(jt)

#保存标记完成的图片
cv2.imwrite("save.png", jt)