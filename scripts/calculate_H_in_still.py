#!/usr/bin/env python
from cv2 import sqrt
import rospy, cv2, cv_bridge
import numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


ARUCO_DICT = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
              "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
              "DICT_5X5_50": cv2.aruco.DICT_5X5_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
              "DICT_5X5_250": cv2.aruco.DICT_5X5_250, "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
              "DICT_6X6_50": cv2.aruco.DICT_6X6_50, "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
              "DICT_6X6_250": cv2.aruco.DICT_6X6_250, "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
              "DICT_7X7_50": cv2.aruco.DICT_7X7_50, "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
              "DICT_7X7_250": cv2.aruco.DICT_7X7_250, "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
              "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
              "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
              "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
              "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
              "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11}

class HCalculate:

        def __init__(self):

                self.bridge = cv_bridge.CvBridge()

                self.image_sub = rospy.Subscriber('camera/image',
                        Image, self.image_callback)

        def image_callback(self, msg):
                global  M,count
                image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                lower_yellow = numpy.array([ 10, 10, 10])
                upper_yellow = numpy.array([255, 255, 250])

                lower_white = numpy.array([0, 0, 80])
                upper_white = numpy.array([180, 43, 220])
                
                mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
                mask2 = cv2.inRange(hsv, lower_white, upper_white)
                mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
                mask4 = cv2.inRange(hsv, lower_white, upper_white)

                h, w, d = image.shape
                search_top = 2*h/3
                serach_top2=2*h/3+79
                mask1[0:search_top, 0:w] = 0
                mask2[0:search_top, 0:w] = 0

                mask3[0:serach_top2,0:w]=0
                mask4[0:serach_top2,0:w]=0
                
                M1 = cv2.moments(mask1)
                M2 = cv2.moments(mask2)
                M3 = cv2.moments(mask3)
                M4 = cv2.moments(mask4)



                if M1['m00'] > 0:
                    cx1 = int(M1['m10']/M1['m00'])
                    cy1 = int(M1['m01']/M1['m00'])

                    cx2 = int(M2['m10']/M2['m00'])
                    cy2 = int(M2['m01']/M2['m00'])

                    cx3 = int(M3['m10']/M3['m00'])
                    cy3 = int(M3['m01']/M3['m00'])

                    cx4 = int(M4['m10']/M4['m00'])
                    cy4 = int(M4['m01']/M4['m00'])

                    point1=numpy.array([[cx1,cy1],[cx2,cy2],[cx3,cy3],[cx4,cy4]],dtype="float32")
                    cy1=int ( cy1-(cy3-cy1)*0.732)
                    cy2=int ( cy2-(cy4-cy2)*0.732)
                    point2=numpy.array([[cx1,cy1],[cx2,cy2],[cx1,cy3],[cx2,cy4]],dtype="float32")

                #     if((cx1+cx2)/2==(cx3+cx4)/2 and count==0):
                    M=cv2.getPerspectiveTransform(point1,point2)
                    rospy.loginfo(type(M))
                    rospy.loginfo(M)
                    

                    fpt_x = (cx1 + cx2)/2
                    fpt_y = (cy1 + cy2)/2 + 2*h/3
                    # rospy.loginfo("%f,%f",cy1,cy3)

                    warped=cv2.warpPerspective(image,M,(w,h))


                    cv2.circle(image, (cx1, cy1), 10, (0,255,255), -1)
                    cv2.circle(image, (cx2, cy2), 10, (255,255,255), -1)
                    cv2.circle(image, (cx3, cy3), 10, (0,255,255), -1)
                    cv2.circle(image, (cx4, cy4), 10, (255,255,255), -1)
                    cv2.circle(image, (fpt_x, fpt_y), 10, (128,128,128), -1)

                cv2.imshow("window", image)
                cv2.imshow("warped",warped)
                
                cv2.waitKey(1)

rospy.init_node('HCaluate')
count=0
HCalculate = HCalculate()
rospy.spin()
