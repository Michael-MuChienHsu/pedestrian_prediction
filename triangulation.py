import pickle
from utils import make_video_from_dir
import numpy as np
import argparse
from triangulation_utils import *
from triangulation_config import triangulation_configuratrion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml_path", type = str, default="triangulation.yaml", help="Path to yaml.")    
    args = parser.parse_args()
    config = triangulation_configuratrion(args.yaml_path)

    # Load 2d joints.
    f = open(config.detected_joint_path, 'rb')
    data = pickle.load(f)
    
    # Set up multiview camera parameter.
    P_list, trajectory = setup_multicamera(config) # trajectory tracklet ground truth trajectory.
    num_tracklet = len(trajectory)

    # Visualize 3d joints over time.
    if config.visualize["visualize"]:
        if config.tracker_id == -1:
            raise ValueError("Visualization does not support -1 tracker_id (all tracklet).")
        visualize_3D_joint_traj(data, config )
        # write visualized 3d joint in to a video.
        write_video_path = os.path.join(config.visualize["visualize_path"], "3d_video.avi") 
        make_video_from_dir(config.visualize["visualize_path"], write_video_path, fps=3)


    # Display MPJPE
    if config.MPJPE:
        if config.tracker_id == -1:
            for _id in range(num_tracklet):
                mpjpe_list = get_n_view_mpjpe(data, config, P_list, _id)
                display_single_tracker_MPJPE(mpjpe_list, config.use_views, _id)

        else:
            mpjpe_list = get_n_view_mpjpe(data, config, P_list, config.tracker_id)
            display_single_tracker_MPJPE(mpjpe_list, config.use_views, config.tracker_id)


    # Triangulate and save output pickles:
    if config.save_3d_joints:
        if config.tracker_id == -1:
            joints_3d_list = []
            mpjpe_list = []

            for tracker_id, traj in enumerate(trajectory):
                _joints_3d = estimate_3D_points(data, config, tracker_id)
                _mpjpe_list = get_n_view_mpjpe(data, config, P_list, tracker_id)
                joints_3d_list.append(_joints_3d )
                mpjpe_list.append( np.array(_mpjpe_list).mean() )
            
            saved_path = save_3d_joints_estimation(joints_3d_list, trajectory, mpjpe_list, config.video_num, config.tracker_id, config.output_path)

        else:
            joints_3d = estimate_3D_points(data, config, config.tracker_id)
            mpjpe_list = get_n_view_mpjpe(data, config, P_list, config.tracker_id)
            mpjpe = np.array(mpjpe_list).mean()
            saved_path = save_3d_joints_estimation([joints_3d], [trajectory], [mpjpe], config.video_num, config.tracker_id, config.output_path)

        # read_3d_joints(saved_path)
