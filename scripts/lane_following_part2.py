#!/usr/bin/env python
import math
from collections import deque
from datetime import datetime
from time import sleep

import cv2
from numpy import vectorize
import cv_bridge
import numpy
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

class Follower:

    def __init__(self):

        self.bridge = cv_bridge.CvBridge()

        self.image_sub = rospy.Subscriber('camera/image',
                                          Image, self.image_callback)

        self.cmd_vel_pub = rospy.Publisher('cmd_vel',
                                           Twist, queue_size=10)

        self.twist = Twist()
        self.velocity = 0.2


    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = numpy.array([20, 100, 100])
        upper_yellow = numpy.array([34, 255, 250])

        lower_white = numpy.array([0, 0, 100])
        upper_white = numpy.array([180, 43, 220])

        mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask2 = cv2.inRange(hsv, lower_white, upper_white)

        h, w, d = image.shape
        search_top = 2*h/3
        serach_top2 = h/4
        mask1[0:search_top, 0:w] = 0
        mask2[0:search_top, 0:w] = 0

        M1 = cv2.moments(mask1)
        M2 = cv2.moments(mask2)

        warped = cv2.warpPerspective(image, M, (w, h))
        hsv2 = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        mask3 = cv2.inRange(hsv2, lower_yellow, upper_yellow)
        mask4 = cv2.inRange(hsv2, lower_white, upper_white)
        mask3[0:0, 0:w] = 0
        mask4[0:0, 0:w] = 0
        M3 = cv2.moments(mask3)
        M4 = cv2.moments(mask4)


        if M3['m00'] and M4['m00'] > 0:
            cx1 = int(M3['m10']/M3['m00'])
            cy1 = int(M3['m01']/M3['m00'])
            cx2 = int(M4['m10']/M4['m00'])
            cy2 = int(M4['m01']/M4['m00'])
            fpt_x = (cx1 + cx2)/2
            fpt_y = (cy1 + cy2)/2 + 2*h/3
            cv2.circle(warped, (cx1, cy1), 10, (0, 255, 255), -1)
            cv2.circle(warped, (cx2, cy2), 10, (255, 255, 255), -1)
            cv2.circle(warped, (fpt_x, fpt_y), 10, (128, 128, 128), -1)

            err = w/2 - fpt_x


            self.velocity = 0.2
            self.twist.linear.x = self.velocity
            self.twist.angular.z = err*3/80
            self.cmd_vel_pub.publish(self.twist)

            cv2.imshow("window", image)
            cv2.imshow("BEV", warped)
            cv2.waitKey(1)


rospy.init_node('lane_follower')
# M=numpy.array([[-7.24044334e-01,-1.33589686e+00 ,2.75194752e+02],
#  [ 5.88035368e-16,-3.09726306e+00 ,5.01191812e+02],
#  [ 1.88257696e-18,-8.36914725e-03 ,1.00000000e+00]])

# M=numpy.array([[-3.32435282e-01,-1.08187888e+00 ,1.76936863e+02],
#  [ 2.59220803e-15,-3.35902542e+00 ,5.53364648e+02],
#  [ 9.18312741e-18,-8.55101324e-03 ,1.00000000e+00]])

#H matrix
M = numpy.array([[-7.24044334e-01, -1.33589686e+00, 2.75194752e+02],
                 [9.80058946e-16, -3.27278896e+00, 5.43142502e+02],
                 [2.34437961e-18, -8.36914725e-03, 1.00000000e+00]])

follower = Follower()
rospy.spin()


