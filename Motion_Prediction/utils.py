import os
import yaml
import numpy as np
import math
from scipy import interpolate

import matplotlib; matplotlib.use('Tkagg')
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches

''' Yaml Parser
'''
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class VisualizeInterface:
    def __init__(self):
        pass
    
    def check_interrupt(self):
        os.system('clear')
        control = input("[viz] ? [y/n]: ")
        if (control == 'n'): exit()
        plt.clf()
        plt.close()

    def prepare_kf_data(self, traj, input_batch):
        self.x = input_batch['x'][0].detach().cpu().numpy()
        self.y = input_batch['y'][0].detach().cpu().numpy()
        self.traj = traj[0].reshape(-1, 60, 5).detach().cpu().numpy()
        self.lane_graph = input_batch['lane_graph'].detach().cpu().numpy()

    def prepare_argo_data(self, traj, input_batch):
        ''' trajectory-related
        '''
        self.x = input_batch['x'][0].detach().cpu().numpy()
        self.y = input_batch['y'][0].detach().cpu().numpy()
        self.traj = traj[0].reshape(-1, 60, 5).detach().cpu().numpy()

        ''' map-related
        '''
        self.lane_graph = input_batch['lane_graph'].detach().cpu().numpy()
        self.lane_polygon = input_batch['lane_polygon'].detach().cpu().numpy()
    
    def kf_matplot(
        self,
        title,
        traj,
        lane_graph,
        x, y
    ):
        fig, axes = plt.subplots()
        axes.set_title(title, fontsize=11)
        axes.set_facecolor(config['plot']['bg']['color']) 
        ''' lane
        ''' 
        color_lg = config['plot']['lane']['color']
        axes.plot(
            lane_graph[..., 0].T, lane_graph[..., 1].T,
            ':',
            color=color_lg,
            linewidth=config['plot']['lane']['lw'],
            alpha=config['plot']['lane']['alpha'],
            zorder=998
        )
        ''' plot history 
        '''
        axes.plot(
            x[0], x[1], 
            color=config['plot']['history']['color'],
            linewidth=config['plot']['history']['lw'],
            zorder=999,
        )
        ''' plot future groundtruth 
        '''
        axes.plot(
            y[0], y[1], 
            color=config['plot']['gt']['color'],
            linewidth=config['plot']['gt']['lw'],
            alpha=config['plot']['gt']['alpha'],
        )
        for i in range(6):
            ''' plot trajectory planning
            '''
            axes.plot(
                traj[i,:,0], traj[i,:,1], 
                '-',
                color=config['plot']['traj']['color'],
                linewidth=config['plot']['traj']['lw'],
                alpha=config['plot']['traj']['alpha'],
                zorder=999,
            )
        plt.axis('equal')
        plt.pause(0)

    def argo_matplot(
        self, 
        title, 
        traj, 
        lane_graph, lane_polygon, 
        x, y,
    ):
        fig, axes = plt.subplots()
        axes.set_title(title, fontsize=11)
        axes.set_facecolor(config['plot']['bg']['color']) 
        ''' plot lane
        '''
        color_lg = config['plot']['lane']['color']
        color_lp = config['plot']['lane_polygon']['color']
        axes.plot(
            lane_graph[..., 0].T, lane_graph[..., 1].T,
            ':',
            color=color_lg,
            linewidth=config['plot']['lane']['lw'],
            alpha=config['plot']['lane']['alpha'],
            zorder=998
        )
        ''' plot lane polygon
        '''
        axes.plot(
            lane_polygon[..., 0].T, lane_polygon[..., 1].T,
            color=color_lp,
            linewidth=config['plot']['lane_polygon']['lw'],
            alpha=config['plot']['lane_polygon']['alpha'],
            zorder=998
        )
        ''' plot history 
        '''
        axes.plot(
            x[0], x[1], 
            color=config['plot']['history']['color'],
            linewidth=config['plot']['history']['lw'],
            zorder=999,
        )
        ''' plot future groundtruth 
        '''
        axes.plot(
            y[0], y[1], 
            color=config['plot']['gt']['color'],
            linewidth=config['plot']['gt']['lw'],
            alpha=config['plot']['gt']['alpha'],
            zorder=999,
        )
        for i in range(6):
            ''' plot trajectory planning
            '''
            axes.plot(
                traj[i,:,0], traj[i,:,1], 
                '-',
                color=config['plot']['traj']['color'],
                linewidth=config['plot']['traj']['lw'],
                alpha=config['plot']['traj']['alpha'],
            )
        plt.axis('equal')
        plt.pause(0)

    def kf_forward(
        self,
        idx,
        input_batch,
        traj, cls
    ):
        self.check_interrupt()
        self.prepare_kf_data(traj, input_batch)
        self.kf_matplot(
            "kung-fu-motion-dataset",
            self.traj,
            self.lane_graph,
            np.transpose(self.x, (1,0)), np.transpose(self.y, (1,0)),
        )

    def argo_forward(
        self,
        idx, dataset, 
        input_batch, model,
        traj, cls
    ):
        self.check_interrupt()
        self.prepare_argo_data(traj, input_batch)
        self.argo_matplot(
            "argo-motion-dataset",
            self.traj, 
            self.lane_graph, self.lane_polygon,
            np.transpose(self.x, (1,0)), np.transpose(self.y, (1,0)),
        )

def path_smoothing(waypoints, custom_num=120):
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    x = x[okay]
    y = y[okay]
    tck, *rest = interpolate.splprep([x,y])
    u = np.linspace(0, 1, num=custom_num)
    smooth = interpolate.splev(u, tck)
    smooth = np.asarray(smooth)
    return smoot_[:, 1::2]

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
