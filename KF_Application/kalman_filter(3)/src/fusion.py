#!/usr/bin/env python

from tkinter import Scale
import rospy
import math, os
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped as Pose
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
# KF coded by yourself
from Kalman_filter import KalmanFilter

class Fusion:
    def __init__(self):
        rospy.Subscriber('/gps', Pose, self.gpsCallback)
        rospy.Subscriber('/radar_odometry', Odometry, self.odometryCallback)
        rospy.Subscriber('/gt_odom', Odometry, self.gtCallback)
        rospy.on_shutdown(self.shutdown)
        self.posePub = rospy.Publisher('/pred', Odometry, queue_size = 10)
        self.KF = None
        self.step = 0   # Record update times

        self.last_odometry_position = np.zeros(2)
        self.last_odometry_angle = 0

        self.gt_list = []
        self.est_list = []

    def shutdown(self):
        print("shuting down fusion.py")

    def gpsCallback(self, data):
        self.step += 1
        # Get GPS data only for 2D (x, y)
        measurement = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        gps_covariance = np.array(data.pose.covariance).reshape(6, 6)[:2,:2]    

        # KF update
        if self.step == 1:
            self.init_KF(measurement[0], measurement[1], 0)
        else:
            self.KF.R = gps_covariance # @De Yu,Hong GPS measurement covariance

            self.KF.update(np.array([measurement[0], measurement[1], 0])) # @De Yu, Hong
        print(f"estimation: {self.KF.x}")

    def odometryCallback(self, data):
        self.step += 1
        # Read radar odometry data from ros msg
        position = data.pose.pose.position # @De Yu, Hong
        odometry_covariance = np.array(data.pose.covariance).reshape(6, -1)[:2,:2]

        # Get euler angle from quaternion
        (roll, pitch, yaw) = euler_from_quaternion([0,0,data.pose.pose.orientation.z,data.pose.pose.orientation.w]) # @De Yu, Hong

        # Calculate odometry difference
        diff = np.array([position.x - self.last_odometry_position[0], position.y - self.last_odometry_position[1]]) # @De Yu,Hong
        diff_yaw = yaw - self.last_odometry_angle # @De Yu, Hong

        # KF predict
        if self.step == 1:
            self.init_KF(position.x, position.y, 0)
        else: 
            self.KF.Q = np.array([[15,0,0],
                                  [0,15,0],
                                  [0,0,15]])

            self.KF.predict([diff[0], diff[1], diff_yaw]) # @De Yu, Hong odom_ pred [delta_x delta_y delta_yaw]
        print(f"estimation: {self.KF.x}")
        self.last_odometry_position = [position.x, position.y] # @De Yu,Hong Update odometry position
        self.last_odometry_angle = yaw  # @De Yu,Hong Update odometry yaw 

        quaternion = quaternion_from_euler(0, 0, self.KF.x[2]) # @De Yu,Hong predict yaw

        # Publish odometry with covariance
        predPose = Odometry()
        predPose.header.frame_id = 'origin'
        predPose.pose.pose.position.x = self.KF.x[0]     # @De Yu,Hong push in predict x
        predPose.pose.pose.position.y = self.KF.x[1]     # @De Yu,Hong push in predict y
        predPose.pose.pose.orientation.x = quaternion[0] # @De Yu,Hong push in predict orientation x
        predPose.pose.pose.orientation.y = quaternion[1] # @De Yu,Hong push in predict orientation y
        predPose.pose.pose.orientation.z = quaternion[2] # @De Yu,Hong push in predict orientation z
        predPose.pose.pose.orientation.w = quaternion[3] # @De Yu,Hong push in predict orientation w
        predPose.pose.covariance = [self.KF.P[0][0], self.KF.P[0][1],0,0,0,0,
                                    self.KF.P[1][0], self.KF.P[1][1],0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0 ]
        self.posePub.publish(predPose)

    def gtCallback(self, data):
        gt_position = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        self.gt_list.append(gt_position)
        if self.KF is not None:
            kf_position = self.KF.x[:2]
            self.est_list.append(kf_position)

    def plot_path(self):
        plt.figure(figsize=(10, 8))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        gt_x, gt_y = zip(*self.gt_list)
        est_x, est_y = zip(*self.est_list)
        plt.plot(gt_x, gt_y, alpha=0.25, linewidth=8, label='Groundtruth path')
        plt.plot(est_x, est_y, alpha=0.5, linewidth=3, label='Estimation path')
        plt.title("KF fusion odometry result comparison")
        plt.legend()
        if not os.path.exists("/home/ee904/SDC/hw4/src/kalman_filter/results"):
            os.mkdir("/home/ee904/SDC/hw4/src/kalman_filter/results")
        plt.savefig("/home/ee904/SDC/hw4/src/kalman_filter/results/result.png")
        plt.show()

    def init_KF(self, x, y, yaw):
        # Initialize the Kalman filter when the first data comes in
        self.KF = KalmanFilter(x = x, y = y, yaw = yaw)
        self.KF.A = np.array([[1,0,0],
                              [0,1,0],
                              [0,0,1]]) 

        self.KF.B = np.array([[1,0,0],
                              [0,1,0],
                              [0,0,1]]) 

if __name__ == '__main__':
    rospy.init_node('kf', anonymous=True)
    fusion = Fusion()
    rospy.spin()
    #fusion.plot_path()
