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

class Follower:

        def __init__(self):

                self.bridge = cv_bridge.CvBridge()

                self.image_sub = rospy.Subscriber('camera/image',
                        Image, self.image_callback)

                self.cmd_vel_pub = rospy.Publisher('cmd_vel',
                        Twist, queue_size=10)

                self.twist = Twist()


        def calculate_lines(self,frame,lines):
            left=[]
            right=[]
            for line in lines:
                x1,y1,x2,y2=line.reshape(4)
                parameters=numpy.polyfit((x1,x2),(y1,y2),1)
                slope=parameters[0]
                y_intercept=parameters[1]
                if slope < 0 :
                    left.append((slope,y_intercept))
                else : 
                    right.append((slope,y_intercept))
                left_avg = numpy.average(left,axis=0)
                right_avg = numpy.average(right,axis=0)
                left_line = self.calculate_coordinate(frame,parameters=left_avg)
                right_line = self.calculate_coordinate(frame, parameters=right_avg)
                return numpy.array([left_line,right_line])

        def calulate_coordinate(self,frame,parameters):
            slope, y_intercept = parameters
            y1 = frame.shape[0]
            y2 = int(y1-150)
            x1 = int((y1-y_intercept)/slope)
            x2 = int((y2-y_intercept)/slope)
            return numpy.array([x1,y1,x2,y2])

        def visualize_lines(self,frame,lines):
            lines_visualize = numpy.zeros_like(frame)
            if lines is not None:
                for x1,y1,x2,y2 in lines:
                    cv2.line(lines_visualize,(x1,y1),(x2,y2),(0,0,255),5)
            return lines_visualize   


                

        def image_callback(self, msg):
                global  M,count
                image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h, w, d = image.shape
                warped=cv2.warpPerspective(image,M,(w,h))
                gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray,(5,5),0)
                canny = cv2.Canny(blur,50,150)
                lines =self.calculate_lines(warped, canny)
                lines_visualize = self.visualize_lines(warped, lines) 
                output = cv2.addWeighted(warped,0.6,lines_visualize,1,0.1)



                lower_yellow = numpy.array([ 10, 10, 10])
                upper_yellow = numpy.array([255, 255, 250])

                lower_white = numpy.array([0, 0, 80])
                upper_white = numpy.array([180, 43, 220])
                
                

                cv2.imshow("window", image)
                cv2.imshow("warped",warped)
                cv2.imshow("gray",gray)
                cv2.imshow("canny",canny)
                cv2.imshow("output",output)
                
                cv2.waitKey(1)


M=numpy.array([[-7.24044334e-01,-1.33589686e+00 , 2.75194752e+02],
 [ 9.80058946e-16 ,-3.27278896e+00 , 5.43142502e+02],
 [ 2.34437961e-18 ,-8.36914725e-03  ,1.00000000e+00]])
rospy.init_node('lane_follower')

count=0
follower = Follower()
rospy.spin()
