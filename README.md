# AUTO-Drive-YOLOV5AndOpencv
前言：这是我机器学习的课程设计，第一块内容是opencv检测车道线；第二块内容就是YOLOv5识别各种车辆和交通灯。 
实现的是在YOLOv5目标检测的基础上增加语义分割头，然后在Cityscapes数据集上进行训练，代码参考的是TomMo23链接如下：TomMao23/multiyolov5: joint detection and semantic segmentation, based on ultralytics/yolov5, (github.com)在此基础上，增加了车道检测并计算车道的曲率半径，同时计算车辆偏离车道中心的距离，也可计算出识别出的车辆距离本车辆的距离。项目主要分为两大块内容：用传统方法即opencv通过图像矫正、二值化、图像变换对二值化图像进行梯度阈值过滤和颜色阈值过滤然后进行二次函数拟合识别出车道线；第二块内容就是YOLOv5识别各种车辆和交通灯。

————————————————

Preface: this is my machine learning course design, the first piece of content is opencv to detect lane lines; the second piece of content is YOLOv5 to recognize various vehicles and traffic lights. 
The implementation is to add semantic segmentation header on the basis of YOLOv5 target detection, and then trained on Cityscapes dataset, the code reference is TomMo23 link is as follows: TomMao23/multiyolov5: joint detection and semantic segmentation, based on ultralytics/yolov5, (github.com) On this basis, lane detection is added and the radius of curvature of the lane is calculated, while the distance of the vehicle deviating from the center of the lane is calculated, and also the distance of the recognized vehicle from this vehicle can be calculated. The project is mainly divided into two main blocks of content: the traditional method that is opencv through image correction, binarization, image transformation of binarized image gradient threshold filtering and color threshold filtering and then quadratic function fitting to identify the lane lines; the second block of content is the YOLOv5 recognition of a variety of vehicles and traffic lights.
————————————————
Code File Description
line.py:For detecting lane lines
