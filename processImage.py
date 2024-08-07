# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:22:29 2017

@author: yang
"""

import os
import cv2
import matplotlib

import utils
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TKAgg')
# script mainly use for drawing the demo picture

cal_imgs = utils.get_images_by_dir('camera_cal')
object_points, img_points = utils.calibrate(cal_imgs, grid=(9, 6))

# test_imgs = utils.get_images_by_dir('test_images')
test_imgs = utils.get_images_by_dir('new_test')

undistorted = []
for img in test_imgs:
    img = utils.cal_undistort(img, object_points, img_points)
    undistorted.append(img)

trans_on_test = []
for img in undistorted:
    src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])
    dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
    M = cv2.getPerspectiveTransform(src, dst)
    trans = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    trans_on_test.append(trans)

thresh = []
binary_wrapeds = []
histogram = []
for img in undistorted:
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=55, thresh_max=100)
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(70, 255))
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    s_thresh = utils.hls_select(img, channel='s', thresh=(160, 255))
    s_thresh_2 = utils.hls_select(img, channel='s', thresh=(200, 240))

    white_mask = utils.select_white(img)
    yellow_mask = utils.select_yellow(img)

    combined = np.zeros_like(mag_thresh)
    #    combined[(x_thresh==1) | ((mag_thresh == 1) & (dir_thresh == 1)) | (s_thresh==1)] = 1
    #    combined[((mag_thresh == 1) & (dir_thresh == 1))] = 1
    combined[((x_thresh == 1) | (s_thresh == 1)) | ((mag_thresh == 1) & (dir_thresh == 1)) | (white_mask > 0) | (
            s_thresh_2 == 1)] = 1

    src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])
    dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
    M = cv2.getPerspectiveTransform(src, dst)
    binary_warped = cv2.warpPerspective(combined, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    hist = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    histogram.append(hist)

    binary_wrapeds.append(binary_warped)

    thresh.append(combined)

plt.figure(figsize=(20, 68))
i = 0
for binary_warped in binary_wrapeds:
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    plt.subplot(2 * len(thresh), 2, 2 * i + 1)
    plt.imshow(binary_warped, cmap='gray')

    plt.subplot(2 * len(thresh), 2, 2 * i + 2)
    i += 1

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

# plt.figure(figsize=(20, 68))
# for i in range(len(thresh)):
#     plt.subplot(2 * len(thresh), 2, 2 * i + 1)
#     #    plt.title('before thresholds')
#     plt.imshow(undistorted[i][:, :, ::-1])
#
#     plt.subplot(2 * len(thresh), 2, 2 * i + 2)
#     #    plt.title('after thresholds')
#     plt.imshow(thresh[i], cmap='gray')
#
#     plt.show()
#    
plt.figure(figsize=(10, 68))
for i in range(len(thresh)):
    plt.subplot(2 * len(thresh), 2, 2 * i + 1)
    plt.title('thresholded_wraped image')
    plt.imshow(thresh[i], cmap='gray')

    # plt.subplot(2*len(thresh),2,2*i+2)
    # plt.title('histogram')
    # plt.plot(histogram[i])
    plt.show()

# undist = utils.cal_undistort(cal_imgs[0],object_points,img_points)
# plt.figure(figsize=(20,10))
# plt.subplot(2,2,1)
# plt.title('before undistorted')
# plt.imshow(cal_imgs[0][:,:,::-1])
#
# plt.subplot(2,2,2)
# plt.title('after undistorted')
# plt.imshow(undist[:,:,::-1])

# test = utils.select_yellow(undistorted[0])
# plt.imshow(test)
# print(test.shape)
