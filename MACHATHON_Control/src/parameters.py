import numpy as np
import cv2


closing_kernel = np.ones((3, 3), np.uint8)
opening_kernel = np.ones((30, 30), np.uint8)

# img_size = (480, 640, 3)
# 352x288
img_size = (352, 288, 3)

img = np.zeros(img_size)
h, w = img.shape[:2]


# w_top = 300
# w_bottom = 640
# h_top = 240
# h_bottom = 480

h_min = 0
h_max = 255
s_min = 100
s_max = 255
v_min = 0
v_max = 40

w_top = 220
w_bottom = 270
h_top = 135
h_bottom = 315

# w_top = 200
# w_bottom = 270
# h_top = 200
# h_bottom = 315

w_m = 144

w_m = img_size[1]/2
src = np.float32([
    (w_m - w_top/2, h_top),
    (w_m - w_bottom/2, h_bottom),
    (w_m + w_bottom/2, h_bottom),
    (w_m + w_top/2, h_top),
])

dst = np.float32([
    [0, 0],
    [0, h],
    [w, h],
    [w, 0],
])
transfrom_mat = cv2.getPerspectiveTransform(src, dst)


# for hsv thresholding (calculate using trackbars in next cells)
# h_min = 0
# h_max = 85
# s_min = 0
# s_max = 255
# v_min = 85
# v_max = 255


# to get red color
# h_min = 0
# h_max = 255
# s_min = 50
# s_max = 255
# v_min = 0
# v_max = 190
