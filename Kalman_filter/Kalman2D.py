from pathlib import Path
import matplotlib.pyplot as plt
import imageio
import numpy as np
import time, os
import random
from numpy.linalg import inv
from turtle import color
import math
from kalman_filter import KalmanFilter

class Virtual_points(object):
    '''
    Create points_num of paths.\n
    Each path include frame_num of points.\n
    Each point include [x, y, vx, vy, id]
    '''
    def __init__(self, points_num, frame_num) -> None:
        self.paths = []
        self.npaths = []
        self.noise = np.random.normal(0, .75, (points_num, frame_num, 2))
        self.noise2 = np.random.normal(0, .1, (points_num, frame_num, 2))
        self.frame_num = frame_num
        self.points_num = points_num
        self.set_virtual_paths()

    def set_virtual_paths(self):
        for i in range(self.points_num):
            # x = 10 * (random.random() - 0.5)
            # y = 10 * (random.random() - 0.5)
            x = float(5 * np.random.normal(0, .25, 1))
            y = float(5 * np.random.normal(0, .5, 1))
            yaw = 0
            print("initial x:", x)
            nx = x
            ny = y
            nyaw = yaw
            vel_x = 2 * (random.random() - 0.5) + 3.0
            vel_y = 0.5
            nvel_x = vel_x
            nvel_y = vel_y
            id = i
            path = []
            npath = []
            for j in range(self.frame_num):
                path.append([x, y, yaw, vel_x, vel_y, id])
                npath.append([nx, ny, nyaw, nvel_x, nvel_y, id])
                x = x + vel_x
                y = y + vel_y
                yaw = math.atan(vel_y/vel_x)
                nx = x + self.noise[i][j][0]
                ny = y + self.noise[i][j][1]
                nyaw = yaw + np.random.normal(0, 0.1)
                vel_x += -0.2
                vel_y = 0.5 + (random.random() - 0.5) * 0.1
                nvel_x = vel_x + self.noise2[i][j][0]
                nvel_y = vel_y + self.noise2[i][j][1]
            self.paths.append(path)
            self.npaths.append(npath)
        print("nosie shape:", self.noise.shape)
        # self.visualize_path()

    def get_virtual_points(self, frame, add_noise = True):
        '''
        Get virtual radar points for testing\n
        each point contains {'pose': [x, y], 'vel': velocity}
        '''
        points = []
        if not add_noise:
            for path in self.paths:
                point = {}
                point['pose'] = [path[frame][0], path[frame][1]]
                point['vel'] = np.sqrt(path[frame][2]**2 + path[frame][3]**2)
                point['velocity'] = [path[frame][2], path[frame][3]]
                points.append(point)
        else:
            for npath in self.npaths:
                point = {}
                point['pose'] = [npath[frame][0], npath[frame][1]]
                point['vel'] = np.sqrt(npath[frame][2]**2 + npath[frame][3]**2)
                point['velocity'] = [npath[frame][2], npath[frame][3]]
                points.append(point)
        return points

    def visualize_path(self):
        plt.figure(figsize=(10, 8))
        plt.ylabel('y')
        plt.xlabel('x')
        plt.grid(True)
        for path in self.paths:
            poses_x = []
            poses_y = []
            for pose in path:
                poses_x.append(pose[0])
                poses_y.append(pose[1])
            plt.plot(poses_x, poses_y, alpha=0.25)
            # plt.scatter(poses_x, poses_y, alpha=0.5, s=2.5)
        for npath in self.npaths:
            poses_x = []
            poses_y = []
            for pose in npath:
                poses_x.append(pose[0])
                poses_y.append(pose[1])
            plt.scatter(poses_x, poses_y, alpha=0.5, s=10)
        plt.title("Paths and measurements of positions")
        plt.show()

def testFilter():

    # Prepare virtual paths
    numOfFrames = 30
    numOfObjects = 1
    dt = 0.1

    print("Creating paths...")
    Path_creator = Virtual_points(points_num = numOfObjects, frame_num = numOfFrames)

    kf = KalmanFilter(x = Path_creator.paths[0][0][0],
                      y = Path_creator.paths[0][0][1],
                      yaw = Path_creator.paths[0][0][2])

    # Test on one path
    X_hist_x = []
    X_hist_y = []
    Z_hist_x = []
    Z_hist_y = []
    for frame in range(30):
        U = Path_creator.npaths[0][frame][2:5]
        Z = Path_creator.npaths[0][frame][0:2]
        kf.predict([U[1], U[2], U[0]])
        K_X, _ = kf.update(Z)
        X_hist_x.append(K_X[0])
        X_hist_y.append(K_X[1])
        Z_hist_x.append(Z[0])
        Z_hist_y.append(Z[1])

    plt.figure(figsize=(10, 8))
    plt.ylabel('y')
    plt.xlabel('x')
    plt.grid(True)
    plt.scatter(Z_hist_x, Z_hist_y, alpha=0.5, s=10, c='r', label='Measurement')
    plt.plot(X_hist_x, X_hist_y, alpha=0.25, label='Filtered path')
    for path in Path_creator.paths:
        poses_x = []
        poses_y = []
        for pose in path:
            poses_x.append(pose[0])
            poses_y.append(pose[1])
        plt.plot(poses_x, poses_y, alpha=0.25, label='Real path')
    plt.title("Kalman filter result")
    plt.legend()
    plt.show()
    print(kf.x)

def main():
    t = testFilter()

if __name__ == '__main__':
    main()
