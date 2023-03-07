#!/usr/bin/env python3
"""
This node is used to get the camera image over wifi and publish it as a ros topic
"""
import matplotlib.pyplot as plt
import cv2

import numpy as np
from PIL import Image

import os
import time
import matplotlib
import matplotlib.pyplot as plt
from parameters import *
from img_funcs import *



def main():


    width = 288
    height = 352

    cv2.namedWindow('trackbars', cv2.WINDOW_NORMAL)
    create_and_set_trackbars()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 3*width, 2*height)
    vid_path = 'lanes.avi'


    vid_path = 'original_q_40_f_1__16_13_40.avi'
    frames_size = 350
    FRAMES_TO_SKIP = 200
    WAIT_TIME = 1000
    # vid_path = 'sss.mkv'
    cap = cv2.VideoCapture(vid_path)
    idx = 0
    prev_lane = np.ones((width, height, 3), dtype=np.uint8)
    prev_lane_2 = np.ones((width, height, 3), dtype=np.uint8)
    curve_list = []
    steer_list = []
    throttle_list = []
    output_vid = cv2.VideoWriter('output_test.avi', cv2.VideoWriter_fourcc(
        *'DIVX'), 20, (3*width, 2*height))

    while idx < frames_size:
        ret, frame = cap.read()

        src, h_min, h_max, s_min, s_max, v_min, v_max = get_trackbars_values()
        frame_mod = add_points(frame, src)
        img_warped = warper(frame, src)
        img_hsv = hls_img(img_warped, h_min, h_max,
                        s_min, s_max, v_min, v_max)
        opening_img = cv2.erode(img_hsv, opening_kernel)
        prev_lane, _ = extract_lane(opening_img, prev_lane)
        curve, res, _ = calculate_curve(prev_lane)
        curve = round(curve, 2)
        steer, throttle = calc_steer_throttle(curve)

        cv2.putText(res,
                    f"curve: {curve}  f = {idx}",
                    # f"lane track. curve:{curve}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                    )
        cv2.putText(frame_mod,
                    f"camera output",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                    )
        cv2.putText(img_warped,
                    f"perspective",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                    )
        cv2.putText(img_hsv,
                    f"HLS thresholding",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                    )
        cv2.putText(opening_img,
                    f"after Erosion",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                    )


        row_1 = np.hstack((frame_mod, img_warped, img_hsv))
        row_2 = np.hstack((res, opening_img, opening_img))
        res = np.vstack((row_1, row_2))
        cv2.imshow('image', res)

        if idx == (frames_size - 1):

            cap = cv2.VideoCapture(vid_path)
            idx = 0
            print('reset')
        idx += 1
        if (idx > FRAMES_TO_SKIP and idx < (frames_size - 1)):
            output_vid.write(res)
        if idx < FRAMES_TO_SKIP:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(WAIT_TIME) & 0xFF == ord('q'):

                break

        curve_list.append(curve)
        steer_list.append(steer)
        throttle_list.append(throttle)

    output_vid.release()
    plot_parameters(curve_list, steer_list, throttle_list)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
