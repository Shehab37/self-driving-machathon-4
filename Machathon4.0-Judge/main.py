import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact_manual
from IPython.display import display

import time
import numpy as np
# pylint: disable=import-error
import cv2
import keyboard
import matplotlib.pyplot as plt
from machathon_judge import Simulator, Judge



class FPSCounter:
    def __init__(self):
        self.frames = []

    def step(self):
        self.frames.append(time.monotonic())

    def get_fps(self):
        n_seconds = 5

        count = 0
        cur_time = time.monotonic()
        for f in self.frames:
            if cur_time - f < n_seconds:  # Count frames in the past n_seconds
                count += 1

        return count / n_seconds

# parameters

img_size = (480, 640, 3)
img = np.zeros(img_size)
h, w = img.shape[:2]

# for perspective transform (calculated using trackbars in next cells)

# w_top = 120
w_top = 170
# w_bottom = 550
w_bottom = 580
h_top = 300
h_bottom = 430

w_m = 320
src = np.float32([
    (w_m - w_top/2, h_top),
    (w_m - w_bottom/2, h_bottom),
    (w_m + w_bottom/2 , h_bottom),
    (w_m + w_top/2, h_top),
])

dst = np.float32([
    [0, 0],
    [0, h],
    [w, h], 
    [w, 0], 
])


# for hsv thresholding (calculate using trackbars in next cells)
h_min = 0
h_max = 255 
s_min = 0
s_max = 16
v_min = 214
v_max = 244



def warper(img ,  src):
    
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST )  # keep same size as input image

    return warped

def unwarper(img , src):
    
    # Compute and apply inverse perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)
    
    return unwarped


def hsv_img(img, h_min , h_max, s_min, s_max, v_min, v_max):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower  = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img_hsv, lower, upper)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask

def extract_lane(img_hsv):
    img_hsv = img_hsv.copy()
    imgray = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        biggest_contour = max(contours, key = cv2.contourArea)
        image_mask = np.zeros_like(img_hsv)
        output = cv2.drawContours(image_mask, [biggest_contour], -1, (255,255,255), -1)
        flag = False
    except:
        flag = True   
        biggest_contour = []
        output = img_hsv.copy()
    return output  ,flag  





def histogram(org_img , split = 1 , plot = False , min_per = 0.4  ,return_img = False ):
    img = org_img.copy()
    if split == 1:
        histogram = np.sum(img, axis=0)[:,0]
    else:
        histogram = np.sum(img[img.shape[0]//split:,:], axis=0)[:,0]

    max_val = np.max(histogram)
    min_val = min_per * max_val
    
    ind_arr = np.where(histogram >= min_val)
    base_point = int(np.average(ind_arr))
    return base_point 



def calculate_curve(hsv_img, plot = False):
    hsv_img = hsv_img.copy()
    curve_mid_point = histogram(hsv_img , return_img=True , plot = False , min_per=0.94  )
    center_point = 320
    diff = center_point - curve_mid_point

    res = hsv_img.copy()
    cv2.circle(res, (center_point, res.shape[0]), 20, (0,0,255), -1)
    cv2.circle(res, (curve_mid_point, res.shape[0]), 20, (0,255,0), -1)
    res_annotated = res.copy()

    return diff , res , res_annotated

def plot_fps(fps_list):
    plt.plot(fps_list , label = "fps")
    plt.title(f'min = {np.min(fps_list)}, mean = {round(np.mean(fps_list) , 2)} , max = {np.max(fps_list)} ')
    plt.show()



def plot_parameters(curve_list ,Diff_error_list, steer_list , throttle_list):
    # first convert all lists to range(-1 , 1)
    curve_list = [x / 320.0 for x in curve_list]
    steer_list = [x  for x in steer_list]
    throttle_list = [x  for x in throttle_list]
    # plot each parameter with different color
    
    # plot the three parameters on different subplots
    fig, axs = plt.subplots(3, figsize=(20, 10))
    axs[0].plot(curve_list , label = "curve" , alpha = 0.6)
    axs[0].plot(Diff_error_list , label = "Diff_error" , alpha = 0.7)
    axs[1].plot(steer_list , label = "steer" , alpha = 0.8)
    axs[2].plot(throttle_list , label = "throttle")
    
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show()


bad_images = []
bad_images_warped = []
prev_steering = [0]
prev_speed = [0]
fps_list = []


def run_car(simulator: Simulator) -> None:
    
    fps_counter.step()

    # Get the image and show it
    img = simulator.get_image()
    img_size = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    temp = img.copy()
    
    fps = fps_counter.get_fps()
    fps_list.append(fps)

    # state = simulator.get_state() # Get the state of the car (steering, velocity)
    state  = [prev_steering[0], prev_speed[0]]
    
    # state_txt = f'steering: {round(state[0], 2)} velocity: {round(state[1], 2)}'
    # cv2.putText(
    #     img,
    #     f"FPS: {fps:.2f}  {state_txt}",
    #     (10, 30),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.8,
    #     (0, 255, 0),
    #     2,
    #     cv2.LINE_AA,
    # )
    
    
    
    wraped_img = warper(img , src) 
    
    
    
    img_hsv  = hsv_img(wraped_img, h_min , h_max, s_min, s_max, v_min, v_max)
    
    
    np.reshape(img_hsv, (img_size[0], img_size[1], 3))
    img_lane_only , bad_imgs_flag = extract_lane(img_hsv)
    
    # if bad_imgs_flag :
    # bad_images.append(img_hsv)
    # if len(bad_images) > 50:
    #     bad_images.pop(0)
    # bad_images_warped.append(wraped_img)
    # if len(bad_images_warped) > 50:
    #     bad_images.pop(0)
        
    # img_lane_only_2 = img_lane_only.copy()
    # img_lane_only_2[0:140 , :] = img_lane_only_2[0:140 , :]*0.5
    # img_lane_only_2[: , 0:140] = img_lane_only_2[: , 0:140]*0.5
    # img_lane_only_2[: , 500:640] = img_lane_only_2[: , 500:640]*0.5
    # img_lane_only = img_lane_only_2

    curve ,curve_unannottated , curve_annottated = calculate_curve(img_lane_only)

    curve_unwarped = unwarper(curve_unannottated, src)
    


    steering , throttle = get_steering_throttle(curve , state[0] , state[1])
    # steering , throttle = get_steering_throttle_modified(curve , state[0] , state[1])
    # steering , throttle = PID_contoller(curve , state[0] , state[1])
    
    
    # add image_with_curve to the original image
    final_img = cv2.addWeighted(temp, 0.6, curve_unwarped, 1, 0)
    
    cv2.putText(
        final_img,
        f"Curve:{curve:.2f} Steering:{steering:.3f} Throttle:{throttle:.3f} ",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )    
                    


    # row_1 =  np.hstack((img , wraped_img  ,img_hsv )) 
    # row_2 =  np.hstack((final_img , curve_unwarped , curve_annottated ))
    # res = np.vstack((row_1, row_2))
    
    
    # res = np.hstack((img, final_img))
    # # create an image with throttle meter as a verical bar and steering meter as half circle
    # meters_img = np.zeros_like(res)
    # cv2.rectangle(meters_img, (meters_img.shape[1]-100, meters_img.shape[0]), (meters_img.shape[1], meters_img.shape[0]-int(throttle/15*meters_img.shape[0])), (255, 0, 0), -1)
    # cv2.ellipse(meters_img, (int(meters_img.shape[1]/2), int(meters_img.shape[0]/2)), (int(meters_img.shape[1]/2.5), int(meters_img.shape[0]/2.5)), 0, -90, int(-steering*180/np.pi - 90), (0, 255, 0), -1)
    # res = np.vstack((res, meters_img))
    # cv2.imshow("image", res)
    
    # add steer and throttle meters on final_img 
    

    
    cv2.imshow("image", final_img)

    
    cv2.waitKey(1)
    


    # simulator.set_car_steering(steering * simulator.max_steer_angle / 4.7)
    # # simulator.set_car_velocity(throttle * 9.4)
        
    # if state[1] <6.5 :
    #     simulator.set_car_velocity(throttle * 9.4)
    # else :
    #     simulator.set_car_velocity(2.7)
    
    
    simulator.set_car_steering(steering)
    simulator.set_car_velocity(throttle )

    prev_steering[0] = steering
    prev_speed[0] = throttle
    curve_list.append(curve)
    steer_list.append(steering)
    throttle_list.append(throttle)

    
    # works good(steering ,v ) --> (7,5)






# PID_STEER_SPACE = np.linspace(0, P_const * MAX_STEER_RADIAN +  D_const * MAX_STEER_RADIAN, 100)
steer_diff_threshold = 1
throttle_diff_threshold = 1
MAX_STEER_RADIAN = 0.5236
steering_space = np.linspace(0, MAX_STEER_RADIAN, 100) 
# throttling_space = np.linspace(11, 5, 100)  
high_steer_space = np.linspace(0.4, MAX_STEER_RADIAN, 100)
high_throttle_space = np.linspace(8, 4, 100)
# Integration = [0]
# Integration_l = [0]
prev_prev_steer = [0]
P_const = 0.83
D_const = 3.45
# P_const = 0.57
I_const = 0.00

# D_const = 2.55
start_counter = [100]

def calculate_steering(curve , prev_steering):
    steering = curve / 320.0
    
    # D_error = prev_steering - prev_prev_steer[0]
    D_error = steering - prev_prev_steer[0]
    prev_prev_steer[0] = steering
    Diff_error_list.append(D_error)

    steering =  (P_const*steering  + D_const * D_error)
    
    
    if abs(curve) < 70 :
        steering = 0.0

    steering = np.interp(steering,(- 1, 1), (- 0.5236, 0.5236))
    return steering


def calculate_throttle(steering , prev_speed):

    throttle = 50
    if np.abs(steering) > 0.2:
        closest_index = np.argmin(np.abs(high_steer_space - abs(steering)))
        throttle = high_throttle_space[closest_index]
        
    return throttle


def get_steering_throttle(curve , prev_steering , prev_speed):
    
    steering = calculate_steering(curve,prev_steering)
    throttle = calculate_throttle(steering , prev_speed)
    return steering , throttle





cv2.namedWindow("image", cv2.WINDOW_NORMAL)
fps_counter = FPSCounter()

judge = Judge(team_code="84S5jIc1x", zip_file_path="solution_05.zip")

# Pass the function that contains your main solution to the judge
judge.set_run_hook(run_car)

# Start the judge and simulation
fps_list = []
curve_list = []
Diff_error_list = []
steer_list = []
throttle_list = []
# make a timer to calculate the run time of the judge
start_time = time.time()

judge.run(send_score=False, verbose=True)

end_time = time.time()

print(f'Total run time: {end_time - start_time - 6} seconds ')
plot_fps(fps_list)
plot_parameters(curve_list,Diff_error_list , steer_list,throttle_list)
# goal --> (30,30)























































