import pickle
from utils import get_param, read_frame, triangulate, make_video_from_dir, to_homo
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml
from yaml.loader import SafeLoader

_RGB_BODY = ( 1, 165/255,0 ) # orange
_RGB_HEAD = (1, 0, 0) # red
_RGB_LEFT = ( 0, 100/255, 0 ) # dark green
_RGB_RIGHT = ( 0, 0, 1 ) # blue

_SKELETON_COLOR =   [_RGB_BODY, _RGB_BODY, _RGB_BODY, _RGB_BODY, 
                     _RGB_HEAD, _RGB_HEAD, _RGB_HEAD, _RGB_HEAD, _RGB_HEAD, _RGB_HEAD, _RGB_HEAD,
                     _RGB_LEFT, _RGB_LEFT, _RGB_LEFT,_RGB_LEFT, 
                     _RGB_RIGHT, _RGB_RIGHT,_RGB_RIGHT,_RGB_RIGHT ]

def get_skeleton_start_end_point(  kp_3d, use_high_conf_filter, mask ):
    """Given 17 joins, return start and end points to connect joints.
    DO NOT MODIFY.

    Args:
        kp_3d: 3x17 inputs
        use_high_conf_filter: bool. Use high conf_mask or not.
        mask: 17x1 bool array.

    Returns:
        start_points: start point for a line to connect 2 joints. 
        end_points: end point for a line to connect 2 joints
    """
    if kp_3d.shape == (3, 17):
        kp_3d = kp_3d.T

    start_points = [5, 5, 6,12,
                    4, 2, 1, 3, 0, 0, 4,
                    5, 7,11,13,
                    6, 8,12,14]
    end_points   = [6,11,12,11,
                    2, 1, 3, 5, 2, 1, 6,
                    7, 9,13,15,
                    8,10,14,16]
    high_conf_idx = np.where( mask )[0].tolist()

    high_conf_edge_points = [ [kp_3d[s_idx], kp_3d[e_idx]] for i,( s_idx, e_idx) in enumerate(zip(start_points, end_points)) 
                              if i in high_conf_idx or not use_high_conf_filter]
    
    start_points, end_points = np.transpose(np.array(high_conf_edge_points), (1, 0, 2)) 
    return np.array(start_points), np.array(end_points)

def show_2d_joint(view_num, sess_num, frame_num, kp, out_path="test_skeleton.png"):
    """Visualize detected 2D joints on frame.

    Args:
        view_num:
        sess_num:
        frame_num:
        kp: 17x2 keypoints in 2d
    """
    image_path =  f"./video_data_n{view_num}/tepper_{sess_num}_frames/frame_{str(frame_num).zfill(4)}.png"
    img = cv2.imread(image_path)
    for i, _x_y in enumerate(kp):
        cv2.circle( img, _x_y, 5, (0, 0, 255),  -1)

    cv2.imwrite(out_path, img)

def visualize_3D_joint_traj(config):
    """
    1. I/O
    Read intrinsic, extrinsic from 2 view: view1 and view2.
    view1 includes {points, K1, extrinsic1} has var name: {x1, K1, extrinsic1 }
    view2 includes {points, K2, extrinsic1} has var name: {x2, K2, extrinsic1 }

    2. Triangulation for initial guess
    camera metrix for view 1 and view 2: P1 = K1@extrinsic1, P2 = K2@extrinsic2
    Solve AX = 0, A = [ x1.cross(P1)
                        x2.cross(P2) ]

    3. Z-axis is upside down, so flip z-axis
    4. Force the standing on ground constraint, align the skeleton's such that its z_min on z=0 plane 
    
    Args:
        config: Configuration.
    Returns:
        None
    """
    view_nums = config.use_views
    sess_num = config.video_num
    tracker_id = config.tracker_id 
    sample_rate = config.visualize["sample_rate"]
    max_frame = config.visualize["max_frame"]
    overlay_frames = config.visualize["overlay_frames"]
    smoothing = config.smoothing["smoothing"]
    max_sliding_window_length = config.smoothing["sliding_window_length"]
    use_high_conf_filter = config.use_high_conf_filter

    if len(view_nums)<2:
        raise ValueError(f"Expect at least 2 views for triangulation, got {len(view_nums)}.")

    P_list = []
    kp_traj = []
    hign_conf_mask = []
    traj_starts = []
    video_starts = []
    for view in view_nums:

        K, k, H, traj_start_frames, start_frame, trajectory, extrinsic = get_param(view_num = view,
                                                                                   video_num = sess_num,
                                                                                   tracker_id = tracker_id)
        P_list.append(K@extrinsic)
        kp, mask = get_2d_joints(data, sess_num, view, tracker_id)
        kp_traj.append(kp.reshape(-1, 2))
        hign_conf_mask.append(mask)
        traj_starts.append(traj_start_frames)
        video_starts.append(start_frame)

    # Process lists to np.ndarrray
    kp_traj = np.array(kp_traj) # V x 17*N x 2
    hign_conf_mask = np.array(hign_conf_mask).reshape( len(hign_conf_mask), -1 ) # V x N*17
    
    # TODO use_high_conf_filter
    # if use_high_conf_filter:
    #     estimated_3d = -np.ones( (kp_traj.shape[1], 3) ) # Buffer, 17*N x 3
    #     estimated_high_conf_3d = triangulate(P_list, [_kp_traj[matching_table] for _kp_traj in kp_traj])
    #     estimated_3d[matching_table] = estimated_high_conf_3d
    # else:
    #     estimated_3d = triangulate(P_list, kp_traj)

    estimated_3d = triangulate(P_list, kp_traj)

    # Magic operation for z axis is upside-down.
    estimated_3d = estimated_3d.reshape(-1, 17, 3)
    estimated_3d = estimated_3d*(np.array( [1, 1, -1] ).reshape(1, 1, 3))
    
    # set figure for visualization
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim([-100, 100])
    ax.set_ylim([-150, -20])
    ax.set_zlim([0, 70])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    color_map = plt.get_cmap("jet")
    # estimated_3d = [estimated_3d[i] for i in range(0, len(estimated_3d), sample_rate)]
    estimated_3d = [estimated_3d[i] for i in range(0, len(estimated_3d), sample_rate)][:max_frame]
    trajectory = [ trajectory[i] for i in range(0, len(trajectory), sample_rate) ]

    # TODO Modulize this part
    # Visualize 2d joints in coresponding 2d view.
    # kp_traj[0] = kp_traj[0].reshape(-1, 17, 2)
    # kp_traj[1] = kp_traj[1].reshape(-1, 17, 2)
    # kp1 = [ kp_traj[0][i] for i in range(0, len(kp_traj[0]), sample_rate) ]
    # kp2 = [ kp_traj[1][i] for i in range(0, len(kp_traj[1]), sample_rate) ]
    # print( traj_starts[0], video_starts[0] )
    # for i, (_kp1, _kp2) in enumerate(zip(kp1, kp2)):
    #     show_2d_joint(1, sess_num, sample_rate*i+traj_starts[0]+video_starts[0], _kp1.astype(int), out_path=f"./view1_skeleton/label_{tracker_id}_frame{i}.png")
    #     show_2d_joint(2, sess_num, sample_rate*i+traj_starts[1]+video_starts[1], _kp2.astype(int), out_path=f"./view2_skeleton/label_{tracker_id}_frame{i}.png")
    #     if i+1 == max_frame: break

    smoothing_offset = 0


    # union_mask: Takes only joints that are detected in all views with high confidence into account
    union_mask = np.sum(hign_conf_mask, axis=0) >= hign_conf_mask.shape[0] # N*17
    matching_table = np.where(union_mask) # N*17
    if smoothing:
        estimated_3d = smooth_3d_joints( estimated_3d, max_sliding_window_length, use_high_conf_filter, union_mask)

    for i, (joints_3d, _mask) in enumerate( zip(estimated_3d, union_mask.reshape(-1, 17))):
        # color = color_map(i/float(len(smoothed_3d_joints)))
        color = (0, 0, 1)

        joints_3d = joints_3d.T
        if False:
            # TODO add use_high_conf_mask to only-z-smooth part    
            # min_z = min(joints_3d[-1][:])
            # max_z = max(joints_3d[-1][:])
            # smoothing_offset = min_z

            # sliding window min_z
            smoothing_offset = np.mean(np.array(estimated_3d[max(0, i-10):min(len(estimated_3d), i+10)])[:, -2:, -1])

            # normalize           
            smoothing_offset = np.array( [ 0, 0, smoothing_offset] ).reshape(3, 1)
            joints_3d = joints_3d - smoothing_offset
            # print(f"person height: {2.54*(max_z-min_z)} cm")

        # plot joint
        if not overlay_frames:
            plt.cla()
            ax.set_xlim([-100, 100])
            ax.set_ylim([-150, -20])
            ax.set_zlim([0, 70])

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        # plot traj at time i 
        ax.scatter( *(trajectory[i]), color = color )

        # plot skeleton
        start_3d, end_3d = get_skeleton_start_end_point(joints_3d, use_high_conf_filter, _mask)
        for _s, _e, _c in zip( start_3d, end_3d, _SKELETON_COLOR ):
            ax.plot([_s[0], _e[0]], [_s[1], _e[1]], [_s[2], _e[2] ], color = _c )

        plt.savefig(f"./joint_3d_visualize/track_frame_{str(i).zfill(3)}.png")
    plt.savefig(f"./joints_3d_{tracker_id}.png")

def estimate_3D_points(config):
    """Apply triangualtion on joints detected on N views to get 3D coordinate.
    Note that the Z-axis is upside down, we flipped the z-axis and force "stand on the ground constraint".

    Args:
        config: configuration

    Returns:
        estimated_3d: N x 17 x 3 joints.
    """
    view_nums = config.use_views
    sess_num = config.video_num
    tracker_id = config.tracker_id 
    smoothing = config.smoothing["smoothing"]
    max_sliding_window_length = config.smoothing["sliding_window_length"]
    use_high_conf_filter = config.use_high_conf_filter

    if len(view_nums)<2:
        raise ValueError(f"Expect at least 2 views for triangulation, got {len(view_nums)}.")

    P_list = []
    kp_traj = []
    hign_conf_mask = []
    for view in view_nums:
        _kp_traj = data[sess_num][view][tracker_id]
        K, k, H, traj_start_frames, start_frame, trajectory, extrinsic = get_param(view_num = view,
                                                                                   video_num = sess_num,
                                                                                   tracker_id = tracker_id)
        P_list.append(K@extrinsic)
        kp, mask = get_2d_joints(data, sess_num, view, tracker_id)
        kp_traj.append(kp.reshape(-1, 2))
        hign_conf_mask.append(mask)

    # Process lists to np.ndarrray
    kp_traj = np.array(kp_traj) # V x 17*N x 2
    hign_conf_mask = np.array(hign_conf_mask).reshape( len(hign_conf_mask), -1 ) # V x N*17

    estimated_3d = triangulate(P_list, kp_traj)
    estimated_3d = estimated_3d.reshape(-1, 17, 3)

    # Flip z axis. If using view 1 and 2
    # if view_nums == [1, 2] or view_nums == [1, 2, 3]:
    #     estimated_3d[:, :, -1] *= -1
    estimated_3d[:, :, -1] *= -1

    if smoothing:
        # union_mask: Takes only joints that are detected in all views with high confidence into account
        union_mask = np.sum(hign_conf_mask, axis=0) >= hign_conf_mask.shape[0] # N*17
        estimated_3d = smooth_3d_joints(estimated_3d, max_sliding_window_length, use_high_conf_filter, union_mask)
   
    return estimated_3d

def smooth_3d_joints( estimated_3d, max_sliding_window_length, use_high_conf_filter, high_conf_mask):
    """Smooth estimated 3d joints using sliding window on their gravity to adjust skeleton's center.

    Args:
        estimated_3d: estimated 3d joints through triangulation. Shape: Nx17x3
        max_sliding_window_length: sliding window length for both sides for smoothing.
        use_high_conf_filter: bool, use high confidence filter.
        high_conf_mask: Bool arraym, high confidence mask, True for keep, False for discard.

    Returns:  
        smoothed_3d_joints: 3d joints after smoothing. Nx17x3
    """
    smoothed_3d_joints = np.empty( (0, 17, 3) )
    estimated_3d = np.array(estimated_3d)
    input_shape = np.array(estimated_3d).shape
    
    if use_high_conf_filter:
        gravity_vector = []
        for _est_joint, _mask in zip( estimated_3d, high_conf_mask.reshape((-1, 17))):
            _est_joint = _est_joint[_mask]
            gravity_vector.append(np.mean(_est_joint, axis=0))
        gravity_vector = np.array(gravity_vector)

    else:
        gravity_vector = np.mean(estimated_3d, axis=1) # Nx3

    for i, (joints_3d, current_gravity) in enumerate(zip(estimated_3d, gravity_vector)):
        sliding_window_length = min( i - max(0, i-max_sliding_window_length), min(i+max_sliding_window_length, len(estimated_3d) ) - i  )
        sliding_window_start = i - sliding_window_length
        sliding_window_end = i + sliding_window_length +1

        average_gravity  = np.mean( gravity_vector[sliding_window_start: sliding_window_end ], axis=0 )
        smoothing_offset = current_gravity - average_gravity

        joints_3d = joints_3d + smoothing_offset.reshape(1, 3)
        smoothed_3d_joints = np.vstack([smoothed_3d_joints, joints_3d[None, :, :]])

    if use_high_conf_filter:
        smoothed_3d_joints_buffer = np.ones((input_shape[0]*17, 3))
        print(smoothed_3d_joints.reshape(-1, 3).shape, smoothed_3d_joints_buffer.shape, smoothed_3d_joints_buffer[high_conf_mask].shape)
        smoothed_3d_joints_buffer[high_conf_mask] = smoothed_3d_joints.reshape(-1, 3)
        smoothed_3d_joints= smoothed_3d_joints_buffer

    return smoothed_3d_joints

def get_2d_joints(kp_dict, sess, view, tracklet_id, threshold = 0.5):
    """Get detected 2d points from dictionary. Filter will keep only keypoints with high conf (>= threshold) and mark low conf.

    Args:
        kp_dict: dict[sess][view][tracklet_id], where kp_dict[sess][view][tracklet_id] is a list. Each element is a dictionary wiht 2 keys:
                "keypoints" for 2d detected 2d keypoits, and "bbox" for detected bounding box.
        sess: video num: 0-7
        view: camera num: 1-3
        tracklet_id: label for tracklet
        threshold: Minimum confidence level for valid joint for triangulation.
        
    Return:
        joints_2d: Nx17x2 np.ndarray. N for trajectory length.
        high_conf_mask: Nx17x1 np.ndarray. True for keep, False for discard.
    """

    traj = kp_dict[sess][view][tracklet_id]
    joints_2d = np.array( [ _traj["keypoints"] for _traj in traj ] )
    high_conf_mask =  (joints_2d[:, :, 2] >= threshold ).reshape(-1, 17 ,1)
    joints_2d = joints_2d[:,:, :2].reshape(-1, 17, 2)
    
    return joints_2d, high_conf_mask

# TODO
def save_3d_joints_estimation(estimated_3d, output_path):
    """Save reconstructed 3D joints to pickle.
    
    Args:
        estimated_3d: 3d joint estimated using triangulation.
        output_path: Path to write pickle file

    Returns:    
        None
    """
    pass

def get_MPJPE(pts_3d, pts_2d, P):
    """Calculate MPJPE: mean per joint position error.
    
    Args:
        pts_3d: joints in 3d space. Nx17x3
        pts_2d: joints in 2d frame. Nx17x2
        P: camera parameter.

    Returns:
        error: MPJPE
    """

    pts_2d = pts_2d.reshape(-1, 2)
    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))]) 
    
    # Since the extrinic for triangulation using view 1 and 2 is upside-down, z = -z
    pts_3d = pts_3d*(np.array( [1, 1, -1, 1] ).reshape(1, 4))

    reprojected_2d_joint = pts_3d@(P.T)
    reprojected_2d_joint = (reprojected_2d_joint/reprojected_2d_joint[:, -1, None])[:, :2]

    diff = pts_2d - reprojected_2d_joint
    norm_diff = np.linalg.norm(diff, axis=1)
    # joint_norm_diff = norm_diff.reshape(-1, 17)

    return np.mean(norm_diff)

def get_n_view_mpjpe(data, config, P_list):
    """Apply triangulation to to 2D point and reprojet back to detected joints in 2D and return MPJPE.

    Args:
        data: Dictionary to get 2d joint position.
        config: configuration.
        P_list: List of camera param to estimate 3D points.
    
    Returns:
        mpjpe_list: List of mpjpes across all view points.
    """
    view_num_list = config.use_views
    sess = config.video_num
    tracker_id = config.tracker_id
    smoothing = config.smoothing

    if not len(view_num_list) == len(P_list):
        raise ValueError(f"Expect same len for view_num_list and P_list, but get {len(view_num_list)} and {len(P_list)}")

    joints_3d = estimate_3D_points(config)

    mpjpe_list = []
    for _view, _P in zip(view_num_list, P_list):
        _joints_2d_view, _= get_2d_joints( data, sess, _view, tracker_id )
        _mpjpe = get_MPJPE(joints_3d, _joints_2d_view, _P)
        mpjpe_list.append( _mpjpe )

    return mpjpe_list

class Config:
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            config = yaml.load(f, Loader=SafeLoader)
        
        self.detected_joint_path = config["pickle_path"]
        self.video_num = config["sess"]
        self.tracker_id = config["tracker_id"]
        self.output_path = config["output_skeleton_path"]
        self.smoothing = config["smoothing"]
        self.use_high_conf_filter = config["use_high_conf_filter"]
        self.use_views = config["use_views"]
        self.visualize = config["visualize"]
        self.MPJPE = config["MPJPE"]

if __name__ == "__main__":
    # python triangulation.py -y "triangulation.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml_path", type = str, default="todo.pickle", help="Path to yaml.")    
    args = parser.parse_args()
    config = Config(args.yaml_path)

    # Load 2d joints.
    f = open(config.detected_joint_path, 'rb')
    data = pickle.load(f)

    # Set up multiview camera matrix.
    K, k, H, traj_start_frames, start_frame, trajectory, extrinsic = get_param(view_num = 1,
                                                                               video_num = config.video_num,
                                                                               tracker_id = config.tracker_id)
    P1 = K@extrinsic

    K, k, H, traj_start_frames, start_frame, trajectory, extrinsic = get_param(view_num = 2,
                                                                               video_num = config.video_num,
                                                                               tracker_id = 0)
    P2 = K@extrinsic

    K, k, H, traj_start_frames, start_frame, trajectory, extrinsic = get_param(view_num = 3,
                                                                               video_num = config.video_num,
                                                                               tracker_id = config.tracker_id)
    P3 = K@extrinsic
    P = [-1, P1, P2, P3]
    P_list = [ P[i] for i in config.use_views ]

    # Visualize 3d joints over time.
    if config.visualize["visualize"]:
        visualize_3D_joint_traj( config )
    else:
        print("false")
    

    if  config.MPJPE:
        mpjpe_list = get_n_view_mpjpe(data, config, P_list)
        print_str = f"label {config.tracker_id}"

        # 3D space scale in inch.
        for _mpjpe,  _view in zip(mpjpe_list, config.use_views):
            print_str += f" view {_view}: {_mpjpe*2.54} cm"
        print(print_str)


    # make_video_from_dir("./joint_3d_visualize/", "./joint_3d_visualize/3d_video.avi", fps=3)