import numpy as np
import pandas as pd
import json

import torch
from torch.utils.data import Dataset, DataLoader

import os
from os import listdir
from os.path import isfile, join
import time
import datetime

from tqdm import tqdm
import yaml
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from utils import path_smoothing, euler_from_quaternion

'''
Matplotlib
'''
matplotlib.use('Tkagg')

''' Yaml Parser
'''
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

'''
Configure constants
'''
_TIME_HORIZON = int(config['constant']['time_horizon'])
_OBS_STEPS = int(config['constant']['obs_steps'])
_PRED_STEPS = int(config['constant']['pred_steps'])
_TOTAL_STEPS = int(config['constant']['total_steps'])
_OBJECT_TYPE = {
    "vehicle": 0,
    "pedestrian": 1,
    "motorcyclist": 2,
    "cyclist": 3,
    "bus": 4,
    "static": 5,
    "background": 6,
    "construction": 7,
    "riderless_bicycle": 8,
    "unknown": 9,
}
_KF_OBJECT_TYPE = {
    "car": 0,
    "pedestrian": 1,
    "bimo": 2,
    "bus": 4,
    "truck": 4,
}
class KungFuDataset(Dataset):
    def __init__(self, tracks_dir, map_dir, processed_dir):
        self.tracks_dir = tracks_dir
        self.map_dir = map_dir
        self.processed_dir = processed_dir
        # tracking 
        self.tracks_list = sorted(self.tracks_dir.rglob("*.json"))
        # lane
        self.parse_lanes()
        
        self.history_list = []
        self.parse_scenario()
        self.sample_idx = 0 
        self.sample_list = []
        self.compute_sample()
    
    def downsample(self, polyline, desire_len):
        index = np.linspace(0, len(polyline)-1, desire_len).astype(int)
        return polyline[index]
    
    def parse_lanes(self):
        lanes_pd = pd.read_json(self.map_dir/Path('waypoints.json'))['waypoints']
        lanes = [] 
        for l in lanes_pd:
            lane = torch.Tensor([[f['x'], f['y'], f['z']] for f in l['points']])
            angle = torch.Tensor([f['angle'] * (180/3.14159) for f in l['points']])
            delta_deg = angle[-1] - angle[0]
            
            ''' turn
            '''
            if delta_deg > 20 and delta_deg < 100:
                lane_split = self.downsample(lane, 10)
                lanes.append(lane_split)
                continue
            elif delta_deg < -20 and delta_deg > -100:
                lane_split = self.downsample(lane, 10)
                lanes.append(lane_split)
                continue
            
            ''' straight
            '''
            splits = lane.shape[0]//10
            for split in range(splits-1):
                lane_split = lane[split*10:(split+1)*10]
                lanes.append(lane_split)
            lane_split = self.downsample(lane[(splits-1)*10:], 10)
            lanes.append(lane_split)
        self.lanes = torch.stack(lanes)
        
    def compute_sample(self):
        for h_idx, history in enumerate(self.history_list):
            if h_idx % 100 != 99 and h_idx % 100 != 49:
                continue
            for f_idx, (track_idx, feature) in enumerate(sorted(history.items())):
                # get agents (-1, +0] tracking
                feature = np.stack(feature)
                feature = torch.from_numpy(feature).to(torch.float32)
                if feature.shape[0] != _TOTAL_STEPS:
                    continue
                if torch.norm(feature[-1,:2]-feature[0,:2]) < 5:
                    continue
                
                sample_path = os.path.join(self.processed_dir, f'data_{self.sample_idx}.pt')
                self.sample_idx += 1
                try:
                    sample = torch.load(sample_path)
                except:
                    sample = {}
                    orig = feature[_OBS_STEPS-1, :2].clone()
                    theta = np.arctan2(feature[_OBS_STEPS-1, 3], feature[_OBS_STEPS-1, 2])
                    rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
                    feature[:, :2] = feature[:, :2] - orig
                    feature[:, :2] = feature[:, :2].mm(rot)
                    feature[:, 2:4] = feature[:, 2:4].mm(rot)
                    feature[:, 4] = feature[:, 4] - theta
                    
                    # get map
                    lanes = self.lanes[..., :2] - orig
                    lanes = lanes.reshape(-1, 2).mm(rot).reshape(-1, 10, 2)
                    lane_features = []
                    for lane in lanes:
                        if torch.norm(lane[0]) > 80:
                            continue
                        lane_features.append(lane) 
                    lane_features = torch.stack(lane_features)
                    
                    # get neighbor
                    n_features = []
                    for n_idx, (n_idx, n_feature) in enumerate(sorted(history.items())):
                        n_feature = np.stack(n_feature)
                        n_feature = torch.from_numpy(n_feature).to(torch.float32)
                        if n_idx == track_idx:
                            continue
                        if n_feature.shape[0] - _PRED_STEPS < 11:
                            continue
                        n_feature[:, :2] = n_feature[:, :2] - orig
                        n_feature[:, :2] = n_feature[:, :2].mm(rot)
                        n_feature[:, 2:4] = n_feature[:, 2:4].mm(rot)
                        n_feature[:, 4] = n_feature[:, 4] - theta
                        
                        size = feature.shape[0]
                        n_features.append(n_feature[size-_PRED_STEPS-11:size-_PRED_STEPS,:])

                    # stack sample
                    sample['x'] = feature[:_OBS_STEPS].reshape(-1, _OBS_STEPS, 6)
                    sample['y'] = feature[_OBS_STEPS:].reshape(-1, _PRED_STEPS, 6)
                    sample['lane_graph'] = lane_features.reshape(-1, 10, 2)
                    sample['neighbor_graph'] = torch.zeros(1, 11, 6) if len(n_features) == 0 else torch.stack(n_features).reshape(-1, 11, 6)
                    torch.save(sample, sample_path)
                self.sample_list.append(sample)
    
    def update_history(self, data, history):
        id_list = [object['tracking_id'] for object in data['objects']]
        for key in list(history.keys()):
            if key not in id_list:
                history.pop(key)
        
        for object in data['objects']:
            id = object['tracking_id']
            rot = object['rotation']
            roll, pitch, yaw = euler_from_quaternion(
                rot['x'], rot['y'], rot['z'], rot['w']
            )
            pose = np.array([
                object['translation']['x'],                            
                object['translation']['y'],                            
                object['velocity']['x'],                            
                object['velocity']['y'],                            
                yaw,
                int(_KF_OBJECT_TYPE[object['tracking_name']])
            ])
            if id not in history.keys():
                history[id] = [pose]
            else:
                history[id].append(pose)
            if len(history[id]) > _TOTAL_STEPS:
                history[id].pop(0)

        self.history_list.append(history)
        return self.history_list[-1].copy() if len(self.history_list) > 0 else {}

    def parse_scenario(self):
        for tracks in self.tracks_list:
            df = pd.read_json(tracks)
            scenario = {}
            last_history = {}
            for f_id, f in enumerate(df['frames']):
                last_history = self.update_history(f, last_history)
    
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        return sample

# function called after collecting samples in batch
def kf_multi_agent_collate_fn(batch):
    elem = [k for k, v in batch[0].items()]
    def _collate_util(input_list, key):
        if isinstance(input_list[0], torch.Tensor):
            return torch.cat(input_list, 0)
        if key == 'lane_graph':
            return torch.cat([torch.cat(inp, 0) for inp in input_list], 0)
        return input_list

    def _get_object_type(input_list):
        return input_list[:,0,-1]

    def _get_idxs(all_list):
        idx = 0
        neighbor_idx = 0
        lane_idx = 0
        lane_idxs = []
        neighbor_idxs = []
        for agent, neighbor, lane in all_list:
            idx += agent.shape[0]
            lane_idx += lane.shape[0]
            neighbor_idx += neighbor.shape[0]
            
            neighbor_idxs.append(neighbor_idx)
            lane_idxs.append(lane_idx)
        return {
                'idxs': [i for i in range(1, idx+1)], 
                'neighbor_idxs': neighbor_idxs, 
                'lane_idxs': lane_idxs, 
        }

    def _get_attention_mask(x_idxs, graph_idxs):
        mask = torch.zeros(x_idxs[-1], graph_idxs[-1])
        a_prev = 0; l_prev=0
        for a_idx, l_idx in zip(x_idxs, graph_idxs):
            mask[a_prev:a_idx, l_prev:l_idx] = 1
            a_prev = a_idx; l_prev = l_idx;
        return mask

    collate = {key: _collate_util([d[key] for d in batch], key) for key in elem}
    collate.update(_get_idxs([(d['x'], d['neighbor_graph'], d['lane_graph']) for d in batch]))
    collate.update({'neighbor_mask':_get_attention_mask(collate['idxs'], collate['neighbor_idxs'])})
    collate.update({'lane_mask':_get_attention_mask(collate['idxs'], collate['lane_idxs'])})
    collate.update({'object_type':_get_object_type(collate['x'])})
    return collate

class Argoverse2Dataset(Dataset):
    def __init__(self, raw_dir, processed_dir, testing_stage=False, visualize=False):
        start_time = time.time()
        from av2.datasets.motion_forecasting import scenario_serialization
        from av2.map.map_api import ArgoverseStaticMap
        from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectState, ObjectType, Track, TrackCategory
        import av2.geometry.polyline_utils as polyline_utils
            
        self.visualize = visualize

        self.TrackCategory = TrackCategory
        self.scenario_serialization = scenario_serialization
        self.avm = ArgoverseStaticMap
        self.polyline_utils = polyline_utils
        self.processed_dir = processed_dir
        self.testing_stage = testing_stage 
        
        self.all_scenario_file_list = sorted(raw_dir.rglob("*.parquet"))

    def __len__(self):
        return len(self.all_scenario_file_list)

    def downsample(self, polyline, desire_len):
        index = np.linspace(0, len(polyline)-1, desire_len).astype(int)
        return polyline[index]
    
    def lane_property(self, lane):
        prop = {}
        init_vec = lane[1] - lane[0]
        init_vec = init_vec / torch.norm(init_vec)
        end_vec = lane[-1] - lane[-2]
        end_vec = end_vec / torch.norm(end_vec)
        prop['init_vec'] = init_vec
        prop['init_heading'] = torch.arctan2(init_vec[1], init_vec[0]) #arc tangent x1/x2
        prop['end_vec'] = end_vec
        prop['end_heading'] = torch.arctan2(end_vec[1], end_vec[0]) #arc tangent x1/x2
        prop['heading_change_rad'] = prop['end_heading'] - prop['init_heading']
        prop['heading_change_deg'] = prop['heading_change_rad'] * (180/3.14159)
        prop['lane_type'] = self.classify_lane(prop['heading_change_deg'])
        return prop

    def classify_lane(self, delta_deg):
        ''' type
        0: turning
        1: straight
        '''
        if delta_deg > 20 and delta_deg < 100:
            lane_type = 0
        elif delta_deg < -20 and delta_deg > -100:
            lane_type = 0
        else:
            lane_type = 1
        return lane_type
    
    def is_turning(self, lane):
        prop = self.lane_property(lane)
        lane_type = self.classify_lane(prop['heading_change_deg'])
        return True if lane_type == 0 else False
 
    def __getitem__(self, idx):
        sample_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
        
        try:
            sample = torch.load(sample_path)
        except:
            ''' Init Sample
            '''
            sample = {}
            ''' Load Scenario from parquet
            '''
            scenario_path = self.all_scenario_file_list[idx]

            scenario_id = scenario_path.stem.split("_")[-1]
            static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
            
            scenario = self.scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
            static_map = self.avm.from_json(static_map_path)
            tracks_df = self.scenario_serialization._convert_tracks_to_tabular_format(scenario.tracks)

            ''' Target Trajectory
            '''
            #[1] Load Target Track
            target_df = tracks_df[tracks_df['object_category']==3] 
            target_id = target_df['track_id'].to_numpy()[0]
            target_traj = torch.as_tensor(target_df[['position_x', 'position_y']].to_numpy()).float() 
            #[2] Rotation Normalization
            velocity = torch.as_tensor(target_df[['velocity_x', 'velocity_y']].to_numpy()).float()[_OBS_STEPS-1]
            theta = np.arctan2(velocity[1], velocity[0])
            rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            #[3] Translation Normalization
            orig = target_traj[_OBS_STEPS-1]
            
            ''' Actors Trajectory
            '''
            target_x_track = []
            target_y_track = []
            neighbor_tracks = []
            for track in scenario.tracks:
                actor_timestep = torch.IntTensor([s.timestep for s in track.object_states])
                observed = torch.Tensor([s.observed for s in track.object_states])
                
                if (_OBS_STEPS-1) not in actor_timestep or (_OBS_STEPS-11) not in actor_timestep:
                    continue
                #add point heading
                actor_state = torch.Tensor(
                    [[object_state.position[0], object_state.position[1], object_state.velocity[0], object_state.velocity[1], np.arctan2(object_state.velocity[1], object_state.velocity[0])] for object_state in track.object_states if object_state.timestep < _TOTAL_STEPS]
                )

                actor_state[:, :2] = actor_state[:, :2] - orig
                actor_state[:, :2] = actor_state[:, :2].mm(rot) #position
                actor_state[:, 2:4] = actor_state[:, 2:4].mm(rot) #velocity
                actor_state[:, 4] = actor_state[:, 4] - theta #heading

                if (track.track_id == target_id):
                    target_y_track.append(actor_state[_OBS_STEPS:])
                    if not self.testing_stage:
                        actor_state = torch.cat([actor_state, torch.empty(_TOTAL_STEPS,1).fill_(_OBJECT_TYPE[track.object_type])], -1)
                    else:
                        actor_state = torch.cat([actor_state, torch.empty(_OBS_STEPS,1).fill_(_OBJECT_TYPE[track.object_type])], -1)
                    target_x_track.append(actor_state[:_OBS_STEPS])
                else:
                    start = actor_timestep == _OBS_STEPS-11
                    start_idx = torch.nonzero(start).item()
                    end = actor_timestep == _OBS_STEPS-1
                    end_idx = torch.nonzero(end).item()
                    
                    actor_state = torch.cat([actor_state[start_idx:end_idx+1], torch.empty(11,1).fill_(_OBJECT_TYPE[track.object_type])], -1)
                    neighbor_tracks.append(actor_state)
            
            ''' Lane Graph
            '''
            speed = torch.norm(velocity)
            map_radius = min(166, max(torch.div(speed*_PRED_STEPS,10, rounding_mode="trunc"), 60))
            lane_segments = static_map.get_nearby_lane_segments(orig.cpu().detach().numpy(), map_radius)
            _lane_centerlines_id = np.array([s.id for s in lane_segments])
            lane_centerlines = np.array([list(static_map.get_lane_segment_centerline(s.id)) for s in lane_segments])
            # Fine-grained preprocessing
            lane_splits = []
            lane_centerlines_id = []
            for idx, l in zip(_lane_centerlines_id, lane_centerlines):
                if self.polyline_utils.get_polyline_length(l) < 10.0:
                    lane_splits.append(torch.Tensor(l))
                    lane_centerlines_id.append(idx)
                elif self.is_turning(torch.Tensor(l)):
                    lane_splits.append(torch.Tensor(l))
                    lane_centerlines_id.append(idx)
                else:
                    interp_lane, interp_pts = self.polyline_utils.interp_polyline_by_fixed_waypt_interval(l, 1.0)
                    splits = interp_pts//10
                    for split in range(splits-1):
                        lane_split = interp_lane[split*10:(split+1)*10]
                        lane_splits.append(torch.Tensor(lane_split))
                        lane_centerlines_id.append(idx)
                    lane_split = self.downsample(interp_lane[(splits-1)*10:], 10)
                    lane_splits.append(torch.Tensor(lane_split))
                    lane_centerlines_id.append(idx)
            lane_centerlines = torch.stack(lane_splits) 
            lane_centerlines = lane_centerlines[...,:2] - orig
            lane_centerlines = lane_centerlines.reshape(-1,2).mm(rot).reshape(-1, 10, 2)
            lane_polygon = torch.stack([torch.Tensor(self.polyline_utils.centerline_to_polygon(c.cpu().detach().numpy(), visualize=False)) for c in lane_centerlines])
            
            ''' Crosswalk Graph
            '''
            crosswalk = static_map.get_scenario_ped_crossings()
            crosswalk_polygon_list = [torch.as_tensor(s.polygon[:,:2], dtype=torch.float) for s in crosswalk]
            if len(crosswalk_polygon_list) > 0: 
                crosswalk_polygon = torch.stack(crosswalk_polygon_list).reshape(-1, 5, 2)
                crosswalk_polygon = crosswalk_polygon[...,:2] - orig
                crosswalk_polygon = crosswalk_polygon.reshape(-1,2).mm(rot).reshape(-1, 5, 2)
            else:
                crosswalk_polygon = torch.zeros(1, 5, 2)
            crosswalk_waypoint_list = [torch.stack([torch.as_tensor(edge, dtype=torch.float) for edge in s.get_edges_2d()]) for s in crosswalk]
            if len(crosswalk_waypoint_list) > 0:
                crosswalk_waypoint = torch.stack(crosswalk_waypoint_list).reshape(-1, 2, 2)
                crosswalk_waypoint = crosswalk_waypoint[...,:2] - orig
                crosswalk_waypoint = crosswalk_waypoint.reshape(-1,2).mm(rot).reshape(-1, 2, 2)
            else:
                crosswalk_waypoint = torch.zeros(1, 2, 2)
            
            ''' Stack One Scenario
            '''
            # sample['x'].shape -> [x,y,vx,vy,heading,type] (50, 6) -> (250)
            sample['x'] = torch.stack(target_x_track).reshape(-1, _OBS_STEPS, 6)
            # sample['y'].shape -> [x,y,vx,vy, heading] (60, 5) -> (240)
            sample['y'] = torch.stack(target_y_track).reshape(-1, _PRED_STEPS, 5)
            # sample['orig'].shape -> (2)
            sample['orig'] = orig
            # sample['rot'].shape -> (2, 2)
            sample['rot'] = rot
            # sample['neighbor_graph'].shape -> (N, 11, 6)
            sample['neighbor_graph'] = torch.stack(neighbor_tracks).reshape(-1, 11, 6) 
            # sample['lane_graph'].shape -> (N, 10, 2)
            sample['lane_graph'] = torch.zeros(1, 10, 2) if lane_centerlines.shape[0] == 0 else lane_centerlines.reshape(-1, 10, 2) 
            # sample['crossawalk_graph'].shape -> (N, 2, 2)
            sample['crosswalk_graph'] = torch.zeros(1, 2, 2) if crosswalk_waypoint.shape[0] == 0 else crosswalk_waypoint.reshape(-1, 2, 2) 
            # sample['crossawalk_polygon'].shape -> (N, 5, 2)
            sample['crosswalk_polygon'] = crosswalk_polygon 
            # sample['lane_polygon'].shape -> (N, 21, 2)
            sample['lane_polygon'] = lane_polygon 
            sample['lane_centerlines_id'] = torch.from_numpy(np.array(lane_centerlines_id))
            
            # config
            sample['scenario_id'] = scenario_id
            sample['target_id'] = target_id
            sample['batch_id'] = idx
            
            torch.save(sample, sample_path)
        return sample

# function called after collecting samples in batch
def argo_multi_agent_collate_fn(batch):
    elem = [k for k, v in batch[0].items()]
    def _collate_util(input_list, key):
        if isinstance(input_list[0], torch.Tensor):
            return torch.cat(input_list, 0)
        if key == 'lane_graph':
            return torch.cat([torch.cat(inp, 0) for inp in input_list], 0)
        return input_list

    def _get_object_type(input_list):
        return input_list[:,0,-1]

    def _get_idxs(all_list):
        idx = 0
        neighbor_idx = 0
        lane_idx = 0
        crosswalk_idx = 0
        neighbor_idxs = []
        lane_idxs = []
        crosswalk_idxs = []
        for agent, neighbor, lane, crosswalk in all_list:
            idx += agent.shape[0]
            neighbor_idx += neighbor.shape[0]
            lane_idx += lane.shape[0]
            crosswalk_idx += crosswalk.shape[0]
            
            neighbor_idxs.append(neighbor_idx)
            lane_idxs.append(lane_idx)
            crosswalk_idxs.append(crosswalk_idx)
        return {
                'idxs': [i for i in range(1, idx+1)], 
                'neighbor_idxs': neighbor_idxs, 
                'lane_idxs': lane_idxs, 
                'crosswalk_idxs': crosswalk_idxs
        }

    def _get_attention_mask(x_idxs, graph_idxs):
        mask = torch.zeros(x_idxs[-1], graph_idxs[-1])
        a_prev = 0; l_prev=0
        for a_idx, l_idx in zip(x_idxs, graph_idxs):
            mask[a_prev:a_idx, l_prev:l_idx] = 1
            a_prev = a_idx; l_prev = l_idx;
        return mask

    collate = {key: _collate_util([d[key] for d in batch], key) for key in elem}
    collate.update(_get_idxs([(d['x'], d['neighbor_graph'], d['lane_graph'], d['crosswalk_graph']) for d in batch]))
    collate.update({'neighbor_mask':_get_attention_mask(collate['idxs'], collate['neighbor_idxs'])})
    collate.update({'lane_mask':_get_attention_mask(collate['idxs'], collate['lane_idxs'])})
    collate.update({'crosswalk_mask':_get_attention_mask(collate['idxs'], collate['crosswalk_idxs'])})
    collate.update({'object_type':_get_object_type(collate['x'])})
    return collate

def test_argo_data_pipeline():
   root = config['argo_data']['root']
   val_dir = Path(root) / Path('raw/validation/')

   processed_val_dir = Path(root) / Path('processed/validation/')
   processed_val_dir.mkdir(parents=True, exist_ok=True)
   
   dataset = Argoverse2Dataset(
           val_dir, 
           processed_val_dir, 
           visualize=False
           )
   dataloader = DataLoader(
           dataset, 
           batch_size=1, 
           collate_fn = argo_multi_agent_collate_fn,
           num_workers=0,
           )
   dataiter = tqdm(dataloader)
   for i, data in enumerate(dataiter):
       plot_argo_scenario(data)

def check_interrupt():
    os.system('clear')
    control = input("[viz] ? [y/n]: ")
    if (control == 'n'): exit()
    plt.clf()
    plt.close()

def plot_argo_scenario(sample):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_facecolor('#2b2b2b')

    ''' History Trajectory '''
    x = sample['x'].reshape(_OBS_STEPS, 6)
    ax.plot(x[:,0],
    	    x[:,1],
    	    "-",
    	    color='g',
    	    alpha=1,
    	    linewidth=1
    	   ) 
    ''' Future Trajectory '''
    y = sample['y'].reshape(_PRED_STEPS, 5)
    ax.plot(y[:,0],
	    y[:,1],
	    "-",
	    color='r',
	    alpha=1,
	    linewidth=1
	   )
    
    ''' Lane Centerline '''
    lane = sample['lane_graph'].reshape(-1, 10, 2)
    for i in range(len(lane)):
     ax.plot(lane[i][:,0],
		lane[i][:,1],
		"--",
		color='w',
		alpha=1,
		linewidth=1
	       )
    ''' neighbor_graph '''
    neighbor_lane = sample['neighbor_graph'].reshape(-1,11,6)
    for i in range(len(neighbor_lane)):
        ax.plot(neighbor_lane[i][:,0],
                neighbor_lane[i][:,1],
                "-",
                color='b',
                alpha=1,
                linewidth=1,
                )
    ''' lane '''
    lane_polygon = sample['lane_polygon'].reshape(-1, 21, 2)
    for i in range(len(lane_polygon)):
        polygon = Polygon(lane_polygon[i], closed=True)
        collection = PatchCollection([polygon],color='dimgray')
        ax.add_collection(collection)
    
    ''' cross walk '''
    crosswalk_polygon = sample['crosswalk_polygon'].reshape(-1,5,2)
    for i in range(len(crosswalk_polygon)):
        polygon = Polygon(crosswalk_polygon[i], closed=True)
        collection = PatchCollection([polygon], color = 'purple', alpha=0.5)
        ax.add_collection(collection)

    plt.axis('equal')
    plt.xlim((-25, 25)) 
    plt.ylim((-25, 25)) 
    plt.show()
    
    check_interrupt()

if __name__ == '__main__':
    test_argo_data_pipeline()
