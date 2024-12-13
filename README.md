# AUTO-Drive-YOLOV5AndOpencv
前言：这是我机器学习的课程设计，第一块内容是opencv检测车道线；第二块内容就是YOLOv5识别各种车辆和交通灯。 
实现的是在YOLOv5目标检测的基础上增加语义分割头，然后在Cityscapes数据集上进行训练，代码参考的是TomMo23链接如下：TomMao23/multiyolov5: joint detection and semantic segmentation, based on ultralytics/yolov5, (github.com)在此基础上，增加了车道检测并计算车道的曲率半径，同时计算车辆偏离车道中心的距离，也可计算出识别出的车辆距离本车辆的距离。项目主要分为两大块内容：用传统方法即opencv通过图像矫正、二值化、图像变换对二值化图像进行梯度阈值过滤和颜色阈值过滤然后进行二次函数拟合识别出车道线；第二块内容就是YOLOv5识别各种车辆和交通灯。

————————————————

Preface: this is my machine learning course design, the first piece of content is opencv to detect lane lines; the second piece of content is YOLOv5 to recognize various vehicles and traffic lights. 
The implementation is to add semantic segmentation header on the basis of YOLOv5 target detection, and then trained on Cityscapes dataset, the code reference is TomMo23 link is as follows: TomMao23/multiyolov5: joint detection and semantic segmentation, based on ultralytics/yolov5, (github.com) On this basis, lane detection is added and the radius of curvature of the lane is calculated, while the distance of the vehicle deviating from the center of the lane is calculated, and also the distance of the recognized vehicle from this vehicle can be calculated. The project is mainly divided into two main blocks of content: the traditional method that is opencv through image correction, binarization, image transformation of binarized image gradient threshold filtering and color threshold filtering and then quadratic function fitting to identify the lane lines; the second block of content is the YOLOv5 recognition of a variety of vehicles and traffic lights.

————————————————

Code File Description（代码文件描述）
line.py:Define a class to receive the characteristics of each line detection（定义一个类来接收每条线路检测的特征）

pipeline.py:The main functions are perspective transformation, thresholding, lane line detection and finally visualisation of a set of images（主要功能是对一组图像进行透视变换、阈值处理、车道线检测，并最终进行可视化）

processImage.py:
Read the video and process it frame by frame.
Perform camera correction, thresholding, and perspective transformation on each frame.
Lane line detection using sliding window method to determine whether the lane line has changed and correct it.
Calculate the curvature of the lane and the distance of the vehicle from the centre of the lane and plot this information onto the image.
Save the processed image as a new video file.

读取视频并进行逐帧处理。
对每帧图像进行相机矫正、阈值化、透视变换。
采用滑动窗口方法进行车道线检测，判断车道线是否发生变化并进行修正。
计算车道的曲率和车辆偏离车道中心的距离，并将这些信息绘制到图像上。
将处理后的图像保存为新的视频文件。

utils.py:Defines and implements a number of functions for this project（定义和实现了本项目的一些函数）
