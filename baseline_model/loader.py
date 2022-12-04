from torch.utils.data import DataLoader
from baseline_model.trajectories import TrajectoryDataset
from triangulation_utils import read_3d_joints

def data_loader(train_config, dataset_config, evaluation=False):
    # If special batch processing needed, then provide a collate_fn
    if evaluation:
        dataset = read_3d_joints(dataset_config.train_dataset_path)
    else:
        dataset = read_3d_joints(dataset_config.train_dataset_path)
    sample_rate = int(dataset_config.fps/train_config.fps)
    if sample_rate < 1: sample_rate = 1

    dset = TrajectoryDataset(
        dataset=dataset,
        obs_len=train_config.obs_len,
        pred_len=train_config.pred_len,
        offset_mode=train_config.offset_mode,
        sample_rate=sample_rate
        )

    loader = DataLoader(
        dset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True
        )

    return dset, loader
