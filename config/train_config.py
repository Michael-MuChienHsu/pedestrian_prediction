import yaml
from yaml.loader import SafeLoader
import os

def get_config(yaml_path):
    """Parse train config and dataset config"""
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    return Train_config(config["train_config"]), Dataset_config(config["dataset_config"])
    
class Dataset_config:
    def __init__(self, config):
        self.fps                = config["fps"]
        self.joint_num          = config["joint_num"]
        self.train_dataset_path = config["train_dataset_path"]
        self.eval_dataset_path  = config["eval_dataset_path"]
        
        # sanity check
        if not os.path.exists(self.train_dataset_path):
            raise ValueError(f"Path error: train_dataset: {self.train_dataset_path} do not exist.")
        if not os.path.exists(self.eval_dataset_path):
            raise ValueError(f"Path error: eval_dataset: {self.eval_dataset_path} do not exist.")

class Train_config:
    def __init__(self, config):
        self.num_epoch   = config["train_config"]
        self.batch_size  = config["batch_size"]
        self.obs_len     = config["obs_len"]
        self.pred_len    = config["pred_len"]
        self.fps         = config["fps"]
        self.num_epoch   = config["num_epoch"]
        self.hidden_size = config["hidden_size"]
        self.offset_mode = config["offset_mode"]
        
        self.log_dir = config["log_dir"]
        self.seed = config["seed"]
