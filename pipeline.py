# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:49:26 2017

@author: yang
"""

import os
import cv2
import matplotlib

import utils
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
import line

matplotlib.use('TkAgg')


def thresholding(img):
    # setting all sorts of thresholds
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=35, thresh_max=150)  # 初始值:thresh_min=10,thresh_max=230,
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 150))
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = utils.hls_select(img, thresh=(180, 255))
    lab_thresh = utils.lab_select(img, thresh=(155, 200))
    luv_thresh = utils.luv_select(img, thresh=(225, 255))

    # Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (
            luv_thresh == 1)] = 1

    return threshholded


def processing(img, object_points, img_points, M, Minv, left_line, right_line):
    # camera calibration 相机矫正, image distortion correction 图像畸变矫正
    undist = utils.cal_undistort(img, object_points, img_points)
    # get the thresholded binary image
    thresholded = thresholding(undist)
    # perform perspective  transform
    thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    # perform detection
    if left_line.detected and right_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(thresholded_wraped,
                                                                                           left_line.current_fit,
                                                                                           right_line.current_fit)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholded_wraped)
    left_line.update(left_fit)
    right_line.update(right_fit)

    # 绘制检测到的直线和信息
    area_img = utils.draw_area(undist, thresholded_wraped, Minv, left_fit, right_fit)
    curvature, pos_from_center = utils.calculate_curv_and_pos(thresholded_wraped, left_fit, right_fit)
    # distance = utils.calculate_position(undist,Minv,left_fit,right_fit)
    result = utils.draw_values(area_img, curvature, pos_from_center)  # draw_values 在图像上写曲率、偏离位置、返回的是写有曲率和偏离位置的图片
    # center_text = "Vehicle is %.3fm " % abs(distance)
    # result = cv2.putText(result,center_text,(100, 150))
    return result


#
#
left_line = line.Line()
right_line = line.Line()
cal_imgs = utils.get_images_by_dir('camera_cal') # 加载相机矫正图像
object_points,img_points = utils.calibrate(cal_imgs,grid=(9,6))  # 相机矫正后的目标点和图像点
M,Minv = utils.get_M_Minv()

project_outpath = 'vedio_out/output_yolov5-v2.mp4'
project_video_clip = VideoFileClip("test_vedio/output_original.avi")
project_video_out_clip = project_video_clip.fl_image(lambda clip: processing(clip,object_points,img_points,M,Minv,left_line,right_line))
project_video_out_clip.write_videofile(project_outpath, audio=False)


# # draw the processed test image
# test_imgs = utils.get_images_by_dir('test_images')
# undistorted = []
# for img in test_imgs:
#     img = utils.cal_undistort(img, object_points, img_points)
#     undistorted.append(img)
#
# result = []
# for img in undistorted:
#     res = processing(img, object_points, img_points, M, Minv, left_line, right_line)
#     result.append(res)
#
# plt.figure(figsize=(20,20),dpi=60)
# for i in range(len(result)):
#     plt.subplot(len(result), 1, i + 1)
#     plt.title('thresholded_wraped image')
#     plt.imshow(result[i][:, :, ::-1])
#     plt.show()
