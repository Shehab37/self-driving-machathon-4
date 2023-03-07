#!/usr/bin/env python3
"""
This node is used to get the camera image over wifi and publish it as a ros topic
"""
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import socket
from typing import List, Optional
import cv2
import rospy
import numpy as np
from PIL import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os
import time
import matplotlib
import matplotlib.pyplot as plt
from parameters import *
from img_funcs import *
from std_msgs.msg import Float32
matplotlib.use('Agg')



class CameraSocket:
    """
    Class to get image from a socket server on the esp

    Parameters
    ----------
    ip_address: str
        IP address of the socket server
    port: int
        Port to connect to the socket server
    """

    def __init__(self, ip_address: str, port: int):
        self.ip_address = ip_address
        self.port = port
        self.buffer_size = 2048
        self.server_buffer_size = 1400

    def get_image(self) -> Optional[np.ndarray]:
        """
        Get image from the socket server

        Returns
        -------
        Optional[np.ndarray]
            The image as a number array (None if no image was received)
        """
        client_socket = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_DGRAM)
        client_socket.sendto(str.encode("0"), (self.ip_address, self.port))
        client_socket.settimeout(0.2)
        try:
            image = self.get_image_unsafe(client_socket)
        except Exception:
            image = None

        client_socket.close()
        return image

    def get_image_unsafe(self, client_socket: socket.socket) -> Optional[np.ndarray]:
        """
        Get image from the socket server
        Note: this can throw an exception

        Returns
        -------
        Optional[np.ndarray]
            The image as a number array (None if no image was received)
        """
        n_bytes = client_socket.recvfrom(self.buffer_size)[0]
        n_frames = int(str(n_bytes)[2:-1])

        all_data: List[bytes] = []
        while True:
            msg_from_server = client_socket.recvfrom(self.buffer_size)[0]
            if (
                len(msg_from_server) == self.server_buffer_size
                and msg_from_server[0] == 255
                and msg_from_server[1] == 216
                and msg_from_server[2] == 255
            ):
                all_data = []
            all_data.append(msg_from_server)
            if (
                len(msg_from_server) < self.server_buffer_size
                and msg_from_server[-1] == 217
                and msg_from_server[-2] == 255
            ):
                data = b"".join(all_data)
                image = None
                if n_frames == len(data):
                    np_img = np.frombuffer(data, dtype=np.uint8)
                    image = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
                return image


def main():
    """
    Main function to launch the ros node
    """

    rospy.init_node("camera_socket", anonymous=True)
    ip_address = rospy.get_param("/camera/IP")
    port = rospy.get_param("/camera/port")
    camera_socket = CameraSocket(ip_address, port)

    throttle_pub = rospy.Publisher("/throttle", Float32, queue_size=1)
    steering_pub = rospy.Publisher("/steering", Float32, queue_size=1)
    image_pub = rospy.Publisher(
        rospy.get_param("/camera/image_topic"), Image, queue_size=1
    )
    rate = rospy.Rate(20)

    prev_steer = -50
    prev_throttle = 0
    prev_curve = 0
    steer_list = []
    throttle_list = []
    curve_list = []
    img_l = []  # extra code

    start_time = time.time()
    # to calculate latency of getting the next frame
    latency_list = []
    latency_t1 = time.time()

    height, width, layers = (352, 288, 3)
    prev_lane_img = np.ones((height, width, 3), dtype=np.uint8)
    curr_time = time.localtime()
    time_str = time.strftime("%H_%M_%S", curr_time)
    original_vid_name = f'original_q_40_f_1__{time_str}.avi'
    vid_original = cv2.VideoWriter(original_vid_name, cv2.VideoWriter_fourcc(
        *'DIVX'), 5, (width, height))

    while not rospy.is_shutdown():
        image = camera_socket.get_image()
        if image is None:
            continue
        image_msg = CvBridge().cv2_to_imgmsg(image, "bgr8")
        image_pub.publish(image_msg)
        rate.sleep()

        img = np.rot90(image)
        img = cv2.resize(img, (width, height))
        img_l.append(img)
        vid_original.write(img)

        curve, prev_lane_img = from_img_to_curve(img, prev_lane_img)

        steer, throttle = calc_steer_throttle(curve, prev_steer, prev_throttle)
        if abs(curve - prev_curve) > 1:
            steer = prev_steer
            throttle = prev_throttle

        send_steer_throttle(steer, throttle, prev_steer,
                            prev_throttle, throttle_pub, steering_pub)
        print(steer, throttle)
        prev_curve = curve
        prev_steer = steer
        prev_throttle = throttle
        curve_list.append(curve)
        steer_list.append(steer)
        throttle_list.append(throttle)
        img_l.append(img)
        latency_t2 = time.time()
        latency_list.append(latency_t2 - latency_t1)
        latency_t1 = latency_t2


    vid_original.release()
    print(
        f'vid_original saved at{os.getcwd()}//{original_vid_name}  with {len(img_l)} frames')

    # make a plot of the latency on matplotlib and save it as a png
    plt.plot(latency_list)
    plt.ylabel('latency')
    plt.xlabel('frame number')
    plt.savefig(f'latency_{original_vid_name}.png')
    print('latency plot saved')
    plot_parameters(curve_list, steer_list, throttle_list)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
