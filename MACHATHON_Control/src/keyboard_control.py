#!/usr/bin/env python3
"""
Module to control the car using a keyboard
"""
import rospy
from std_msgs.msg import Float32
import numpy as np

from sshkeyboard import listen_keyboard
# when key is pressed it will call the callback function
# when key is hold it will call the callback function every 0.1 seconds
# from sshkeyboard import listen_keyboard_hold

# throttle_space = [-255, 255]
# throttle_space = np.linspace(-255, 255, 10)
throttle_space = [-250, -200, -150, -100, -60, 0, 60, 100, 150, 200, 250]


class KeyboardControl:
    """
    Class to control the car using a keyboard

    Parameters
    ----------
    throttle_pub: rospy.Publisher
        Publisher to publish the throttle value
    steering_pub: rospy.Publisher
        Publisher to publish the steering value
    """

    def __init__(self, throttle_pub: rospy.Publisher, steering_pub: rospy.Publisher):
        self.throttle_pub = throttle_pub
        self.steering_pub = steering_pub
        self.throttle = 0
        self.steering = 0
        self.current_throttle_index = 5
        self.current_steer_index = 5

    def key_pressed(self, key: str) -> None:
        """
        Callback for when a key is pressed

        Parameters
        ----------
        key : str
            The key that was pressed
        """
        if key == "w":
            self.throttle = 120
        if key == "s":
            self.throttle = -120
        if key == "d":
            self.steering = -10
        if key == "a":
            self.steering = 10
        if key == "q":
            self.throttle = 0
            self.steering = 0
        # self.steering -= 5
        print(f'steering = {self.steering}, throttle = {self.throttle}')
        self.publish()

    def key_pressed_2(self, key: str) -> None:

        if key == "w":
            self.current_throttle_index += 1
            self.throttle = throttle_space[self.current_throttle_index]
        if key == "s":
            self.current_throttle_index -= 1
            self.throttle = throttle_space[self.current_throttle_index]
        if key == "d":
            self.steering += 50
        if key == "a":
            self.steering += -50
        if key == "q":
            self.throttle = 0
            self.steering = 0
        print(f'steering = {self.steering}, throttle = {self.throttle}')
        self.publish()

    def publish(self):
        """
        Publish the throttle and steering values
        """
        self.throttle_pub.publish(self.throttle)
        self.steering_pub.publish(self.steering)


def main():
    """
    Main function
    """
    rospy.init_node("keyboard_control")

    throttle_pub = rospy.Publisher("/throttle", Float32, queue_size=1)
    steering_pub = rospy.Publisher("/steering", Float32, queue_size=1)

    keyboard_control = KeyboardControl(throttle_pub, steering_pub)

    listen_keyboard(on_press=keyboard_control.key_pressed)
    # listen_keyboard(on_press=keyboard_control.key_pressed_2)
    # listen_keyboard_hold(on_press=keyboard_control.key_pressed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
