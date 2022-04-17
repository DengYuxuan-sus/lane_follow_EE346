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

desired_aruco_dictionary = "DICT_6X6_50"

# The different ArUco dictionaries built into the OpenCV library.
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

# dist=numpy.array(([[-0.58650416 , 0.59103816, -0.00443272 , 0.00357844 ,-0.27203275]]))
# newcameramtx=numpy.array([[189.076828   ,  0.    ,     361.20126638]
#  ,[  0 ,2.01627296e+04 ,4.52759577e+02]
#  ,[0, 0, 1]])
# mtx=numpy.array([[398.12724231  , 0.      ,   304.35638757],
#  [  0.       ,  345.38259888, 282.49861858],
#  [  0.,           0.,           1.        ]])

dist = numpy.array(([[-0.2909375306628219, 0.05890305811963341,
                   0.002023707366213156, 0.002460957243230047, 0]]))
newcameramtx = numpy.array([[189.076828,  0.,     361.20126638], [
                           0, 2.01627296e+04, 4.52759577e+02], [0, 0, 1]])
mtx = numpy.array([[164.595714370777, 0, 155.7296974048595],
                  [0, 165.5485348916819, 108.2763701447475],
                   [0.,           0.,           1.]])


class Follower:

    def __init__(self):

        self.bridge = cv_bridge.CvBridge()

        self.image_sub = rospy.Subscriber('camera/image',
                                          Image, self.image_callback)

        self.cmd_vel_pub = rospy.Publisher('cmd_vel',
                                           Twist, queue_size=10)

        self.twist = Twist()
        self.distance = 0
        self.previous_err = 0
        self.last_err = 0
        self.velocity = 0.6
        self.err_deque = deque()

    def aruco_detect(self, image):
        # global distance
        this_aruco_dictionary = cv2.aruco.Dictionary_get(
            ARUCO_DICT[desired_aruco_dictionary])
        this_aruco_parameters = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
            image, this_aruco_dictionary, parameters=this_aruco_parameters)

        if len(corners) > 0:
            # rospy.loginfo("the distance is %f",distance)
            ids = ids.flatten()
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.05, mtx, dist)
            (rvec-tvec).any()
            for i in range(rvec.shape[0]):
                cv2.aruco.drawAxis(
                    image, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.05)
                cv2.aruco.drawDetectedMarkers(image, corners)
            cv2.putText(image, "Id: " + str(ids), (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            deg = rvec[0][0][2]/math.pi*180
            R = numpy.zeros((3, 3), dtype=numpy.float64)
            cv2.Rodrigues(rvec, R)
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            if not singular:
                x = math.atan2(R[2, 1], R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0
            rx = x * 180.0 / 3.141592653589793
            ry = y * 180.0 / 3.141592653589793
            rz = z * 180.0 / 3.141592653589793

            cv2.putText(image, 'deg_z:'+str(ry)+str('deg'), (0, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            self.distance = ((tvec[0][0][2] + 0.02) * 0.0254) * 100

            cv2.putText(image, 'distance:' + str(round(self.distance, 4)) + str('m'),
                        (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return len(corners) > 0

    def shutdown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")
        self.twist.linear.x = 0
        self.twist.angular.z = 0
        self.cmd_vel_pub.publish(self.twist)

    def image_callback(self, msg):
        global count
        kp = 0.035
        kd =0.025

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

        detect = self.aruco_detect(image)

        if M3['m00'] and M4['m00'] > 0:
            # if M4['m00']==0:
            #         M4['m00']=0.01
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

            if detect and 0.9<self.distance <10 and count < 1:
                    if self.distance<1.1:
                        self.shutdown()
                        start_time = datetime.now()
                        rospy.sleep(5)
                        rospy.loginfo("stop")
                        end_time = datetime.now()
                        durn = (end_time-start_time).total_seconds()
                        rospy.loginfo("Total time =%.4f", durn)
                        count += 1
                    else:
                             self.velocity=0.3
                             self.twist.linear.x = self.velocity
                             self.twist.angular.z = err*3/80
                             self.cmd_vel_pub.publish(self.twist)
            else:

                self.twist.linear.x = 0.3
                self.twist.angular.z = err*3/80
                self.cmd_vel_pub.publish(self.twist)
                
                if not detect:
                        count=0

            cv2.imshow("window", image)
            cv2.imshow("up", warped)

            cv2.waitKey(1)


rospy.init_node('lane_follower')
# M=numpy.array([[-7.24044334e-01,-1.33589686e+00 ,2.75194752e+02],
#  [ 5.88035368e-16,-3.09726306e+00 ,5.01191812e+02],
#  [ 1.88257696e-18,-8.36914725e-03 ,1.00000000e+00]])

# M=numpy.array([[-3.32435282e-01,-1.08187888e+00 ,1.76936863e+02],
#  [ 2.59220803e-15,-3.35902542e+00 ,5.53364648e+02],
#  [ 9.18312741e-18,-8.55101324e-03 ,1.00000000e+00]])

M = numpy.array([[-7.24044334e-01, -1.33589686e+00, 2.75194752e+02],
                 [9.80058946e-16, -3.27278896e+00, 5.43142502e+02],
                 [2.34437961e-18, -8.36914725e-03, 1.00000000e+00]])

count = 0
distance = 0
follower = Follower()
rospy.spin()


