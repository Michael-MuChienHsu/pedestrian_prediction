import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, dataset, obs_len=8, pred_len=12, offset_mode=True, sample_rate = 24):
        super(TrajectoryDataset, self).__init__()
        self.dataset = dataset # list of dict with keys: 3d_joints, 3d_trajectoy, error
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_len = obs_len + pred_len

        self.offset_mode = offset_mode
        self.sample_rate = sample_rate
        self.seq_num = len(self.dataset)
        print(f'Total number of trajectories: {self.seq_num}')
        self._process_valid_dataset()
        
        print(f'Total number of valid trajectories: {self.seq_num}')

    def _process_valid_dataset(self):
        valid_dataset_indice = []
        for i in range(self.seq_num):
            if self.dataset[i]["3d_joints"].shape[0]//self.sample_rate >= self.traj_len:
                valid_dataset_indice.append(i)
        self.dataset = self.dataset[valid_dataset_indice]
        self.seq_num = len(self.dataset)


    def _align_offset(self, estimated_joint, trajectory):
        """Align trajectory to the relative position but keep first frame absolute."""
        estimated_joint[1:] = estimated_joint[1:] - estimated_joint[0:-1]
        trajectory[1:] = trajectory[1:] - trajectory[0:-1]
        return estimated_joint, trajectory


    def _extract_segment(self, estimated_joint, trajectory):
        """Randomly extract self.traj_len continuous frames from tracklet's trajectory, and subsample continuopus
        frame with self.sample_rate.

        Args:
            estimated_joint: full length joint trajectory. N x 17 x 3
            trajectory: full length trajectory.  N x 3

        Returns:
            estimated_joint: Selected valid joint segment self.traj_len x 17 x 3
            trajectory:  Selected valid trajectory segment self.traj_len x 3
        """
        
        seq_traj_len = estimated_joint.shape[0]
        least_frame_length = (self.traj_len-1)*self.sample_rate+1
        start_frame = np.random.randint(0, seq_traj_len-least_frame_length)
        estimated_joint, trajectory = estimated_joint[start_frame:], trajectory[start_frame:]

        estimated_joint = estimated_joint[::self.sample_rate][:self.traj_len]
        trajectory = trajectory[::self.sample_rate][:self.traj_len]

        return estimated_joint, trajectory
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        estimated_joint = self.dataset[index]["3d_joints"]
        trajectory = self.dataset[index]["3d_trajectoy"][:, :2] #use only x, y.
        estimated_joint, trajectory = self._extract_segment(estimated_joint, trajectory )
        if self.offset_mode:
            estimated_joint, trajectory = self._align_offset(estimated_joint, trajectory)
        
        estimated_joint = torch.from_numpy(estimated_joint).type(torch.float32)
        trajectory = torch.from_numpy(trajectory).type(torch.float32)

        return estimated_joint, trajectory