import pickle
from utils import make_video_from_dir
import numpy as np
import argparse
import yaml
from yaml.loader import SafeLoader
from triangulation_utils import *

class Config:
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            config = yaml.load(f, Loader=SafeLoader)
        
        self.video_num = config["sess"]
        self.detected_joint_path = f"./3d_extracted_joints/{self.video_num}.pickle"
        self.tracker_id = config["tracker_id"]
        self.save_3d_joints = config["save_3d_joints"]
        self.output_path = config["output_skeleton_path"]
        self.smoothing = config["smoothing"]
        self.use_high_conf_filter = config["use_high_conf_filter"]
        self.use_views = config["use_views"]
        self.visualize = config["visualize"]
        self.MPJPE = config["MPJPE"]

        # Sanity Check.
        if 0 in self.use_views:
            raise ValueError(f"Use_views are 1-indexed, expect list without 0, but got {self.use_views}")

if __name__ == "__main__":
    # python triangulation.py -y "triangulation.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml_path", type = str, default="triangulation.yaml", help="Path to yaml.")    
    args = parser.parse_args()
    config = Config(args.yaml_path)

    # Load 2d joints.
    f = open(config.detected_joint_path, 'rb')
    data = pickle.load(f)

    # Set up multiview camera parameter.
    P_list, num_tracklet = setup_multicamera(config)

    # Visualize 3d joints over time.
    if config.visualize["visualize"]:
        if config.tracker_id == -1:
            raise ValueError("Visualization does not support -1 tracker_id (all tracklet).")
        visualize_3D_joint_traj( config )
    else:
        print("false")

    # Display MPJPE
    if config.MPJPE:
        if config.tracker_id == -1:
            for _id in range(num_tracklet):
                mpjpe_list = get_n_view_mpjpe(data, config, P_list, _id)
                display_single_tracker_MPJPE(mpjpe_list, config.use_views, _id)

        else:
            mpjpe_list = get_n_view_mpjpe(data, config, P_list, config.tracker_id)
            display_single_tracker_MPJPE(mpjpe_list, config.use_views, config.tracker_id)

    # make_video_from_dir("./joint_3d_visualize/", "./joint_3d_visualize/3d_video.avi", fps=3)

    # Triangulate and save output pickles:
    if config.save_3d_joints:
        joints_3d = estimate_3D_points(data, config)
        save_3d_joints_estimation(joints_3d, config.output_path, config.tracker_id)