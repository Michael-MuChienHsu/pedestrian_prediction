import yaml
from yaml.loader import SafeLoader
import os

class triangulation_configuratrion:
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            config = yaml.load(f, Loader=SafeLoader)
        
        self.video_num            = config["sess"]
        self.detected_joint_path  = f"./3d_extracted_joints/{self.video_num}.pickle"
        self.tracker_id           = config["tracker_id"]
        self.save_3d_joints       = config["save_3d_joints"]
        self.output_path          = config["output_skeleton_path"]
        self.smoothing            = config["smoothing"]
        self.use_high_conf_filter = config["use_high_conf_filter"]
        self.use_views            = config["use_views"]
        self.visualize            = config["visualize"]
        self.MPJPE                = config["MPJPE"]
        self.iterations           = config["iterations"]

        # Sanity Check.
        if 0 in self.use_views:
            raise ValueError(f"Use_views are 1-indexed, expect list without 0, but got {self.use_views}")
            
        if not os.path.exists(self.visualize["visualize_path"]) and self.visualize["visualize"]:
            write_path = self.visualize["visualize_path"]
            print(f"Visualization output path {write_path} not exist, make one.")
            os.mkdir(self.visualize["visualize_path"])
