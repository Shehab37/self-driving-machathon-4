import numpy as np
import cv2
from parameters import *
import matplotlib.pyplot as plt


def nothing(x):
    pass


def create_and_set_trackbars():
    cv2.createTrackbar('h_min', 'trackbars', 0, 255, nothing)
    cv2.createTrackbar('h_max', 'trackbars', 0, 255, nothing)
    cv2.createTrackbar('s_min', 'trackbars', 0, 255, nothing)
    cv2.createTrackbar('s_max', 'trackbars', 0, 255, nothing)
    cv2.createTrackbar('v_min', 'trackbars', 0, 255, nothing)
    cv2.createTrackbar('v_max', 'trackbars', 0, 255, nothing)

    # now trackbars for perspective (only src) w_top , w_bottom, h_top, h_bottom
    cv2.createTrackbar('w_top', 'trackbars', 0, 640, nothing)
    cv2.createTrackbar('w_bottom', 'trackbars', 0, 640, nothing)
    cv2.createTrackbar('h_top', 'trackbars', 0, 480, nothing)
    cv2.createTrackbar('h_bottom', 'trackbars', 0, 480, nothing)

    # set trackbars to default values
    cv2.setTrackbarPos('h_min', 'trackbars', h_min)
    cv2.setTrackbarPos('h_max', 'trackbars', h_max)
    cv2.setTrackbarPos('s_min', 'trackbars', s_min)
    cv2.setTrackbarPos('s_max', 'trackbars', s_max)
    cv2.setTrackbarPos('v_min', 'trackbars', v_min)
    cv2.setTrackbarPos('v_max', 'trackbars', v_max)

    cv2.setTrackbarPos('w_top', 'trackbars', w_top)
    cv2.setTrackbarPos('w_bottom', 'trackbars', w_bottom)
    cv2.setTrackbarPos('h_top', 'trackbars', h_top)
    cv2.setTrackbarPos('h_bottom', 'trackbars', h_bottom)


def get_trackbars_values():
    # get trackbars values to use them in the main loop
    h_min = cv2.getTrackbarPos('h_min', 'trackbars')
    h_max = cv2.getTrackbarPos('h_max', 'trackbars')
    s_min = cv2.getTrackbarPos('s_min', 'trackbars')
    s_max = cv2.getTrackbarPos('s_max', 'trackbars')
    v_min = cv2.getTrackbarPos('v_min', 'trackbars')
    v_max = cv2.getTrackbarPos('v_max', 'trackbars')

    w_top = cv2.getTrackbarPos('w_top', 'trackbars')
    w_bottom = cv2.getTrackbarPos('w_bottom', 'trackbars')
    h_top = cv2.getTrackbarPos('h_top', 'trackbars')
    h_bottom = cv2.getTrackbarPos('h_bottom', 'trackbars')
    w_m = 144

    src = np.float32([
        (w_m - w_top/2, h_top),
        (w_m - w_bottom/2, h_bottom),
        (w_m + w_bottom/2, h_bottom),
        (w_m + w_top/2, h_top),
    ])
    return src, h_min, h_max, s_min, s_max, v_min, v_max


def add_points(img, src):
    img2 = np.copy(img)
    color = [255, 0, 0]  # Red
    thickness = -1
    radius = 15
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
    cv2.circle(img2, (int(x0), int(y0)), radius, color, thickness)
    cv2.circle(img2, (int(x1), int(y1)), radius, color, thickness)
    cv2.circle(img2, (int(x2), int(y2)), radius, color, thickness)
    cv2.circle(img2, (int(x3), int(y3)), radius, color, thickness)

    return img2


def add_lines(img, src):
    img2 = np.copy(img)
    color = [255, 0, 0]  # Red
    thickness = 2
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
    cv2.line(img2, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
    cv2.line(img2, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    cv2.line(img2, (int(x2), int(y2)), (int(x3), int(y3)), color, thickness)
    cv2.line(img2, (int(x3), int(y3)), (int(x0), int(y0)), color, thickness)
    return img2


def warper(img,  src):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    # keep same size as input image

    return warped


def unwarper(img, src):

    # Compute and apply inverse perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(
        img, Minv, img_size, flags=cv2.INTER_NEAREST)

    return unwarped


def hsv_img(img, h_min, h_max, s_min, s_max, v_min, v_max):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img_hsv, lower, upper)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # result = cv2.bitwise_and(img, img, mask=mask)
    # hstack = np.hstack((img, mask, result))
    # return hstack
    return mask


def hls_img(img, h_min, h_max, s_min, s_max, v_min, v_max):
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img_hls, lower, upper)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # result = cv2.bitwise_and(img, img, mask=mask)
    # hstack = np.hstack((img, mask, result))
    # return hstack
    return mask


def histogram(org_img, split=1, plot=False, min_per=0.8, return_img=False):
    img = org_img.copy()
    # bottom_half = img[img.shape[0]//2:,:]
    # histogram = np.sum(bottom_half, axis=0)
    if split == 1:
        histogram = np.sum(img, axis=0)[:, 0]
    else:
        histogram = np.sum(img[img.shape[0]//split:, :], axis=0)[:, 0]

    max_val = np.max(histogram)
    min_val = min_per * max_val

    ind_arr = np.where(histogram >= min_val)
    base_point = int(np.average(ind_arr))
    return base_point
    # if plot:
    #     plt.figure(figsize=(5,5))
    #     plt.imshow(img)
    #     plt.show
    #     plt.figure(figsize=(5,5))
    #     plt.plot(histogram , color = 'r')
    #     plt.show()
    # # plot the histogram

    # if return_img:
    #     img_hist = np.zeros_like(img)
    #     for x , intensity in enumerate(histogram):
    #         cv2.line(img_hist, (x, img.shape[0]), (x, img.shape[0] - int(intensity)//255//split), (255,0,0), 1)
    #         cv2.circle(img_hist, (base_point, img.shape[0]), 10, (0,0,255), -1)
    #     return base_point , histogram , img_hist

    # return base_point , histogram


def calculate_curve(hsv_img, plot=False):
    hsv_img = hsv_img.copy()

    # center_point ,_, bottom_hist_img = histogram(hsv_img , split = 4, return_img=True , plot = True )
    # curve_mid_point ,_, full_hist_img = histogram(hsv_img , split = 1, return_img=True , plot = True )
    # center_point= histogram(hsv_img , split = 4, return_img=True , plot = True  , min_per=0.3)
    curve_mid_point = histogram(
        hsv_img, return_img=True, plot=False, min_per=0.8)
    # track_mid_point = histogram(hsv_img , return_img=True , plot = False , min_per=0.5 , split = 4  )
    center_point = int(hsv_img.shape[1]/2 - 1)
    diff = center_point - curve_mid_point
    # curve_list.append(diff)
    # if len(curve_list) > 5:
    #     curve_list.pop(0)

    # avg_curve = sum(curve_list) / len(curve_list)

    # if plot:
    #     plt.imshow(full_hist_img)
    #     plt.show()
    #     plt.imshow(bottom_hist_img)
    #     plt.show()

    # print(f'base point is {center_point} and track_mid_point is {curve_mid_point}')
    # print(f'difference is {center_point - curve_mid_point}')

    # add full_hist_img and bottom_hist_img to the original image with different colors to see the difference
    # and add the base_point , track_mid_point to the image
    # base point is blue and curve_mid_point is red and the difference is green and bottom_hist_img is yellow and full_hist_img is purple
    # res = cv2.addWeighted(hsv_img, 1, full_hist_img, 0.5, 0)
    # res = cv2.addWeighted(res, 1, bottom_hist_img, 0.5, 0)
    res = hsv_img.copy()
    cv2.circle(res, (center_point, res.shape[0]-10), 20, (0, 0, 255), -1)
    cv2.circle(res, (curve_mid_point, res.shape[0]-10), 20, (0, 200, 0), -1)
    # cv2.circle(res, (track_mid_point, res.shape[0]), 10, (0,0,0), -1)
    cv2.line(res, (center_point, res.shape[0]),
             (curve_mid_point, res.shape[0]), (0, 200, 0), 5)
    res_annotated = res.copy()
    cv2.putText(res_annotated,
                f"center_point: {center_point} curve_point: {curve_mid_point} diff: {diff/320}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
                )
    # red_points.append(curve_mid_point)
    # blue_points.append(center_point)
    # black_points.append(track_mid_point)
    # if plot:
    #     plt.figure(figsize = (5,5))
    #     plt.imshow(res)
    #     plt.show
    # return avg_curve , res
    diff = round(diff/center_point, 2)
    return diff, res, res_annotated


def extract_lane(img_hsv, prev_lane):
    img_hsv = img_hsv.copy()
    # _,thresh = cv2.threshold(img_hsv, 1, 255, cv2.THRESH_BINARY)
    imgray = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(f'len of contours is {len(contours)}')
    # print(len(contours))
    # now get the biggest contour

    # biggest_contour = max(contours, key = cv2.contourArea)
    try:
        # biggest_contour = max(contours, key = cv2.contourArea)
        # instead of getting the biggest contour we will get the contour that is closest to the previous contour
        results_and = [cv2.bitwise_and(prev_lane, cv2.drawContours(np.zeros_like(
            img_hsv), [contour], -1, (255, 255, 255), -1)) for contour in contours]
        results_or = [cv2.bitwise_or(prev_lane, cv2.drawContours(np.zeros_like(
            img_hsv), [contour], -1, (255, 255, 255), -1)) for contour in contours]
        intersections = [np.sum(x[:, :, 1]) for x in results_and]
        unions = [np.sum(x[:, :, 1]) for x in results_or]
        int_over_unions = [x/y for x, y in zip(intersections, unions)]
        idx = np.argmax(int_over_unions)
        # print(idx)
        # print(f'number of contours is {len(contours)} and the biggest contour is {idx} + results {[np.sum(x[:,:,1]) for x in results]}')

        contour = contours[idx]
        image_mask = np.zeros_like(img_hsv)
        output = cv2.drawContours(
            image_mask, [contour], -1, (255, 255, 255), -1)
        flag = False
    except:
        flag = True
        output = img_hsv
    return output, flag


def curve_from_original_image(img, prev_lane_img, src, h_min, h_max, s_min, s_max, v_min, v_max, visual=False):
    # convert to gray scale
    wraped_img = warper(img, src)
    img_hsv = hsv_img(wraped_img, h_min, h_max, s_min, s_max, v_min, v_max)
    curr_lane_img, bad_imgs_flag = extract_lane(img_hsv, prev_lane_img)
    curve, res, res_annotated = calculate_curve(curr_lane_img)
    if visual == True:
        return curve, wraped_img, img_hsv, curr_lane_img, res_annotated
    return curve, curr_lane_img


def calc_steer_throttle(curve, prev_steer=0, prev_velocity=0):
    # curve is between -1 and 1
    # steer is (-20 , 0 ,20)
    # throttle is (0 , 100 , 150 , 200 ,250)
    steer = prev_steer
    if abs(curve) < 0.05:
        steer = 0
        throttle = 250
    # elif abs(curve) < 0.2:
    #     # steer = 0
    #     throttle = 250
    elif abs(curve) < 0.15:
        # steer = 0
        throttle = 150
    else:
        steer = 20 * np.sign(curve)
        throttle = 100
    # print(f'steer: {steer}  throttle: {throttle}')
    return steer, throttle


def plot_parameters(curve_list, steer_list, throttle_list):
    # first convert all lists to range(-1 , 1)
    # curve_list = [x / 100.0 for x in curve_list]
    # Diff_error_list = [x * 3.0 for x in Diff_error_list]
    # steer_list = [x  for x in steer_list]
    # throttle_list = [x  for x in throttle_list]
    # plot each parameter with different color

    # plot the three parameters on different subplots
    fig, axs = plt.subplots(3, figsize=(20, 10))
    axs[0].plot(curve_list, label="curve", alpha=0.6)
    axs[1].plot(steer_list, label="steer", alpha=0.8)
    # axs[1].plot(Integ_error_list , label = "Integ_error_list" , alpha = 0.5)
    axs[2].plot(throttle_list, label="throttle")

    # show grid lines on all subplots
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    # show grid on the plots
    plt.show()


def from_img_to_curve(img, prev_lane_img):
    img_size = (img.shape[1], img.shape[0])
    # steps to get the curve
    # hsv , closing , warping , lane extraction , curve calculation
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img_hsv, lower, upper)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img_hsv = mask
    img_closed = cv2.morphologyEx(img_hsv, cv2.MORPH_CLOSE, closing_kernel)
    warped = cv2.warpPerspective(img_closed,
                                 transfrom_mat,
                                 img_size, flags=cv2.INTER_NEAREST)
    imgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # biggest_contour = max(contours, key = cv2.contourArea)
    try:
        # biggest_contour = max(contours, key = cv2.contourArea)
        # instead of getting the biggest contour we will get the contour that is closest to the previous contour
        results = [cv2.bitwise_and(prev_lane_img, cv2.drawContours(np.zeros_like(
            warped), [contour], -1, (255, 255, 255), -1)) for contour in contours]
        idx = np.argmax([np.sum(x[:, :, 1]) for x in results])
        # print(f'number of contours is {len(contours)} and the biggest contour is {idx} + results {[np.sum(x[:,:,1]) for x in results]}')

        contour = contours[idx]
        image_mask = np.zeros_like(warped)
        curr_lane = cv2.drawContours(
            image_mask, [contour], -1, (255, 255, 255), -1)
    except:
        curr_lane = warped

    histogram = np.sum(curr_lane, axis=0)[:, 0]
    max_val = np.max(histogram)
    min_val = 0.8 * max_val

    ind_arr = np.where(histogram >= min_val)
    curve_mid_point = int(np.average(ind_arr))

    center_point = img.shape[1] / 2 - 1
    diff = center_point - curve_mid_point
    result = round(diff/center_point, 2)
    return result, curr_lane
