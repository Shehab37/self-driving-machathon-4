#!/usr/bin/env python3
"""
Node to control the car over wifi
"""
import socket
from typing import List, Optional

import rospy
import numpy as np
from std_msgs.msg import Float32


class ControlSocket:
    """
    Parameters
    ----------
    ip_address: str
        IP address of the socket server
    port: int
        Port to connect to the socket server
    timeout: float
        Timeout for the socket
    """

    def __init__(self, ip_address: str, port: int, timeout: float = 0.2):
        self.ip_address = ip_address
        self.port = port
        self.timeout = timeout
        self.buffer_size = 2048
        self.server_buffer_size = 1400
        self.steering = 0
        self.throttle = 0

    def send_action(self) -> Optional[float]:
        """
        Sends the current throttle and steering values to the car
        and returns the speed

        Returns
        -------
        Optional[float]
            The image as a number array (None if no image was received)
        """
        try:
            # Modify the values to be positive to be able to send them as unsigned bytes
            throttle = self.throttle + 255
            steering = self.steering + 180
            to_send = [
                ord("0"),  # Command to send the action
                steering >> 8,
                steering & 255,
                throttle >> 8,
                throttle & 255,
            ]

            client_socket = socket.socket(
                family=socket.AF_INET, type=socket.SOCK_DGRAM)
            client_socket.sendto(
                bytes(to_send),
                (self.ip_address, self.port),
            )
            client_socket.settimeout(self.timeout)
            speed_bytes = client_socket.recvfrom(self.buffer_size)[0]
            speed = float(str(speed_bytes)[2:-1])

            client_socket.close()
            return speed

        except socket.timeout:
            return None

    def set_steering(self, steering_msg: Float32) -> None:
        """
        Set the steering angle

        Parameters
        ----------
        steering_msg : Float32
            The steering angle in degrees
        """
        self.steering = int(steering_msg.data)

    def set_throttle(self, throttle_msg: Float32) -> None:
        """
        Set the throttle value

        Parameters
        ----------
        throttle_msg : Float32
            The throttle value between -255, 255
        """
        self.throttle = int(throttle_msg.data)


def main():
    """
    Main Ros listener loops
    """
    rospy.init_node("car_control")

    ip_address = rospy.get_param("/ESP/IP")
    port = rospy.get_param("/ESP/port")
    steering_topic = rospy.get_param("/ESP/steering_topic")
    throttle_topic = rospy.get_param("/ESP/throttle_topic")
    speed_topic = rospy.get_param("/ESP/speed_topic")
    car_control = ControlSocket(ip_address, port, timeout=0.2)

    rospy.Subscriber(steering_topic, Float32, car_control.set_steering)
    rospy.Subscriber(throttle_topic, Float32, car_control.set_throttle)
    velocity_pub = rospy.Publisher(speed_topic, Float32, queue_size=1)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        speed = car_control.send_action()
        if speed is None:
            continue
        velocity_pub.publish(speed)

        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
