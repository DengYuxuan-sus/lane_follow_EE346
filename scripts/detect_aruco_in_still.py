#!/usr/bin/env python
from cv2 import sqrt
import rospy, cv2, cv_bridge
import numpy,math
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


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

dist=numpy.array(([[-0.2909375306628219, 0.05890305811963341, 0.002023707366213156, 0.002460957243230047, 0]]))
newcameramtx=numpy.array([[189.076828   ,  0.    ,     361.20126638]
 ,[  0 ,2.01627296e+04 ,4.52759577e+02]
 ,[0, 0, 1]])
mtx=numpy.array([[164.595714370777, 0, 155.7296974048595],
 [  0, 165.5485348916819, 108.2763701447475],
 [  0.,           0.,           1.        ]])



class Aruco_detect:

        def __init__(self):

                self.bridge = cv_bridge.CvBridge()

                self.image_sub = rospy.Subscriber('camera/image',
                        Image, self.image_callback)

                self.cmd_vel_pub = rospy.Publisher('cmd_vel',
                        Twist, queue_size=10)

                self.twist = Twist()

        def image_callback(self, msg):
                    global  M,count
                    image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[desired_aruco_dictionary])
                    this_aruco_parameters = cv2.aruco.DetectorParameters_create()
                    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, this_aruco_dictionary, parameters=this_aruco_parameters)

                    if len(corners) > 0:
                                rospy.loginfo("yes")
                                ids = ids.flatten()
                                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
                                (rvec-tvec).any()
                                for i in range(rvec.shape[0]):
                                        cv2.aruco.drawAxis(image, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.05)
                                        cv2.aruco.drawDetectedMarkers(image, corners)
                                cv2.putText(image, "Id: " + str(ids), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)        
                                deg=rvec[0][0][2]/math.pi*180
                                R=numpy.zeros((3,3),dtype=numpy.float64)
                                cv2.Rodrigues(rvec,R)
                                sy=math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
                                singular=sy< 1e-6
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

                                cv2.putText(image,'deg_z:'+str(ry)+str('deg'),(0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                                distance = ((tvec[0][0][2] + 0.02) * 0.0254) * 100  
                                cv2.putText(image, 'distance:' + str(round(distance, 4)) + str('m'), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                 # Loop over the detected ArUco corners
                                # for (marker_corner, marker_id) in zip(corners, ids):
                                        
       
                                #         # Extract the marker corners
                                #         corners = marker_corner.reshape((4, 2))
                                #         (top_left, top_right, bottom_right, bottom_left) = corners


                                #         # Convert the (x,y) coordinate pairs to integers
                                #         top_right = (int(top_right[0]), int(top_right[1]))
                                #         bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                                #         bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                                #         top_left = (int(top_left[0]), int(top_left[1]))
         
                                #         # Draw the bounding box of the ArUco detection
                                #         cv2.line(image, top_left, top_right, (0, 255, 0), 2)
                                #         cv2.line(image, top_right, bottom_right, (0, 255, 0), 2)
                                #         cv2.line(image, bottom_right, bottom_left, (0, 255, 0), 2)
                                #         cv2.line(image, bottom_left, top_left, (0, 255, 0), 2)
         
                                #         # Calculate and draw the center of the ArUco marker
                                #         center_x = int((top_left[0] + bottom_right[0]) / 2.0)
                                #         center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                                #         cv2.circle(image, (center_x, center_y), 4, (0, 0, 255), -1)
         
                                #         # Draw the ArUco marker ID on the video frame
                                #         # The ID is always located at the top_left of the ArUco marker
                                #         cv2.putText(image, str(marker_id), (top_left[0], top_left[1] - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                #         rospy.loginfo("here")
                    else: rospy.loginfo("no")
                    cv2.imshow("window", image)
                    cv2.waitKey(1)

rospy.init_node('Aruco_detect')
count=0
Aruco =Aruco_detect()
rospy.spin()
