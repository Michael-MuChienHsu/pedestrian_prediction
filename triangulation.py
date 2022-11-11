import pickle
from utils import get_param, read_frame, triangulate, make_video_from_dir, to_homo
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_skeleton_start_end_point(  kp_3d ):
    """Given 17 joins, return start and end points to connect joints

    Args:
        kp_3d: 3x17 inputs

    Returns:
        start_points: start point for a line to connect 2 joints. 
        end_points: end point for a line to connect 2 joints
    """
    if kp_3d.shape == (3, 17):
        kp_3d = kp_3d.T

    start_points = [ kp_3d[4], kp_3d[2], kp_3d[1], kp_3d[3], kp_3d[0], kp_3d[0], kp_3d[5], kp_3d[5], kp_3d[7], 
                     kp_3d[5], kp_3d[6], kp_3d[8], kp_3d[6], kp_3d[12],kp_3d[12],kp_3d[14],kp_3d[11],kp_3d[13], kp_3d[4] ]

    end_points =   [ kp_3d[2], kp_3d[1], kp_3d[3], kp_3d[5], kp_3d[2], kp_3d[1], kp_3d[6], kp_3d[7], kp_3d[9], 
                     kp_3d[11],kp_3d[8], kp_3d[10],kp_3d[12],kp_3d[11],kp_3d[14],kp_3d[16],kp_3d[13],kp_3d[15], kp_3d[6] ]

    return np.array(start_points), np.array(end_points)


def show_2d_joint(view_num, sess_num, frame_num, kp):
    """Visualize detected 2D joints on frame.

    Args:
        view_num:
        sess_num:

    """
    image_path =  f"./video_data_n{view_num}/tepper_{sess_num}_frames/frame_{str(frame_num).zfill(4)}.png"
    img = cv2.imread(image_path)
    min_x = min(kp[:, 0])
    min_y = min(kp[:, 1])
    # text_x  = (kp[:, 0] - min_x)*4 + min_x 
    # text_y  = (kp[:, 1] - min_y)*4 + min_y 
    
    for i, _x_y in enumerate(kp):
        cv2.circle( img, _x_y, 5, (0, 0, 255),  -1)
        
        # cv2.putText(img, f"{i}", (text_x[i], text_y[i]), font, fontScale, 
        #                     fontColor, thickness, lineType)        
    cv2.imwrite(f"test_skeleton_{view_num}.png", img)

def visualize_3D_joint_traj(view_nums, sess_num, tracker_id, sample_rate=20, max_frame = 10, smoothing = True, 
                            max_sliding_window_length = 5):
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
        view_nums: list of views (camera) used to triabgulate. e.g. [1, 2, 3].
        sess_num: video number, 0-7.
        tracker_id: label of the tracklet.
        sample_rate: Sample rate to reduce fps for visualization.
        max_frame: Total frames to visualize.
        smoothing: Smooth the joint positions to prevent jittering
        max_sliding_window_length: sliding window average window size 
                                   For frame i, average( i-max_sliding_window_length: i+max_sliding_window_length)

    Returns:
        None
    """
    if len(view_nums)<2:
        raise ValueError(f"Expect at least 2 views for triangulation, got {len(view_nums)}.")

    P_list = []
    kp_traj = []
    for view in view_nums:
        _kp_traj = data[sess_num][view][tracker_id]
        K, k, H, traj_start_frames, start_frame, trajectory, extrinsic = get_param(view_num = view,
                                                                                   video_num = sess_num,
                                                                                   tracker_id = tracker_id)
        P_list.append(K@extrinsic)
        kp_traj.append(np.array([ traj["keypoints"][:, :2] for traj in _kp_traj ]).reshape((-1, 2)))

    estimated_3d = triangulate(P_list, kp_traj)
    estimated_3d = estimated_3d.reshape(-1, 17, 3)
    estimated_3d = estimated_3d*(np.array( [1, 1, -1] ).reshape(1, 1, 3))
    # estimated_3d = estimated_3d[:, :, [1, 0, 2]]

    # set figure for visualization
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # ax.set_xlim([-100, 100])
    # ax.set_ylim([-150, -20])
    # ax.set_zlim([0, 70])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    color_map = plt.get_cmap("jet")
    estimated_3d = [estimated_3d[i] for i in range(0, len(estimated_3d), sample_rate)][:max_frame]
    trajectory = [ trajectory[i] for i in range(0, len(trajectory), sample_rate) ]

    smoothing_offset = 0

    if smoothing:
        smoothed_3d_joints = smooth_3d_joints( estimated_3d, max_sliding_window_length )

    for i, (joints_3d) in enumerate(smoothed_3d_joints):
        color = color_map(i/float(len(smoothed_3d_joints)))

        joints_3d = joints_3d.T       
        if False:
            # min_z = min(joints_3d[-1][:])
            # max_z = max(joints_3d[-1][:])
            # smoothing_offset = min_z
            # sliding window min_z
            # smoothing_offset = np.mean(np.array(estimated_3d[max(0, i-10):min(len(estimated_3d), i+10)])[:, -2:, -1])
            # smoothing_offset = np.array( [ 0, 0, smoothing_offset] ).reshape(3, 1)

            joints_3d[-1][:] = joints_3d[-1][:] - min_z
            print(f"person height: {2.54*(max_z-min_z)} cm")

        # plot joint
        # plt.cla()
        # ax.set_xlim([-100, 100])
        # ax.set_ylim([-150, -20])
        # ax.set_zlim([0, 70])

        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")

        X, Y, Z = joints_3d
        ax.scatter(X, Y, Z, color = color )
        ax.scatter( *(trajectory[i]), color = color )
        # plot skeleton
        start_3d, end_3d = get_skeleton_start_end_point(joints_3d)
        for _s, _e in zip( start_3d, end_3d ):
            ax.plot([_s[0], _e[0]], [_s[1], _e[1]], [_s[2], _e[2] ], color = color )

        plt.savefig(f"./joint_3d_visualize/track_frame_{str(i).zfill(3)}.png")
    plt.savefig(f"./joints_3d_{tracker_id}.png")

    # Mean joint average loss
    # show_2d_joint(view_num_1, sess_num, traj_start_frames1+start_frame1, kp1.astype(int))
    # show_2d_joint(view_num_2, sess_num, traj_start_frames2+start_frame2, kp2.astype(int))

    # reprojected1 = P1.dot( np.vstack( [X, Y, Z, np.ones_like(X)] ) ).T
    # reprojected1 = (reprojected1/reprojected1[:, -1, None])[:, :2]

    # reprojected2 = P2.dot( np.vstack( [X, Y, Z, np.ones_like(X)] ) ).T
    # reprojected2 = (reprojected2/reprojected2[:, -1, None])[:, :2]

    # for (true_x1, true_y1), (true_x2, true_y2), (_x1, _y1), (_x2, _y2) in zip( kp1, kp2, reprojected1, reprojected2 ):
    #     print(f"kp1 {true_x1, true_y1}, repro kp1 {_x1, _y1}, kp2 {true_x2, true_y2}, repro kp2 {_x2, _y2}")

def estimate_3D_points(view_nums, sess_num, tracker_id, sample_rate=1, smoothing = True, max_sliding_window_length = 5):
    """Apply triangualtion on joints detected on N views to get 3D coordinate.
    Note that the Z-axis is upside down, we flipped the z-axis and force "stand on the ground constraint".

    Args:
        view_nums: list of views (camera) used to triabgulate. e.g. [1, 2, 3].
        sess_num: video number, 0-7.
        tracker_id: label of whom tracked.
        sample_rate: the origin video is in 60 FPS, use sample rate to lower fps.

    Returns:
        estimated_3d: N x 17 x 3 joints.
    """
    if len(view_nums)<2:
        raise ValueError(f"Expect at least 2 views for triangulation, got {len(view_nums)}.")

    P_list = []
    kp_traj = []
    for view in view_nums:
        _kp_traj = data[sess_num][view][tracker_id]
        K, k, H, traj_start_frames, start_frame, trajectory, extrinsic = get_param(view_num = view,
                                                                                   video_num = sess_num,
                                                                                   tracker_id = tracker_id)
        P_list.append(K@extrinsic)
        kp_traj.append(np.array([ traj["keypoints"][:, :2] for traj in _kp_traj ]).reshape((-1, 2)))

    estimated_3d = triangulate(P_list, kp_traj)
    estimated_3d = estimated_3d.reshape(-1, 17, 3)

    # Flip z axis. If using view 1 and 2
    if view_nums == [1, 2]:
        estimated_3d[:, :, -1] *= -1

    if smoothing:
        estimated_3d = smooth_3d_joints(estimated_3d, max_sliding_window_length)

    # DEBUG Test output skeleton to ensure the shape is correct.
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # color_map = plt.get_cmap("jet")

    # start_3d, end_3d = get_skeleton_start_end_point(estimated_3d[0].T)
    # X, Y, Z = estimated_3d[0].T

    # color = color_map(6/float(len(estimated_3d)))
    # ax.scatter(X, Y, Z, color = color )
    # for _s, _e in zip( start_3d, end_3d ):
    #     ax.plot([_s[0], _e[0]], [_s[1], _e[1]], [_s[2], _e[2] ], color = color )

    # plt.savefig(f"./test_3d_track_frame_{0}.png")
   
    return estimated_3d

def smooth_3d_joints( estimated_3d, max_sliding_window_length):
    """Smooth estimated 3d joints using sliding window on their gravity to adjust skeleton's center.

    Args:
        estimated_3d: estimated 3d joints through triangulation. Shape: Nx17x3
        max_sliding_window_length: sliding window length for both sides for smoothing.

    Returns:  
        smoothed_3d_joints: 3d joints after smoothing. Nx17x3
    """
    smoothed_3d_joints = np.empty( (0, 17, 3) )
    prev_gravity = 0

    for i, joints_3d in enumerate(estimated_3d):
        current_gravity = np.mean( joints_3d, axis=0 )
        sliding_window_length = min( i - max(0, i-max_sliding_window_length), min(i+max_sliding_window_length, len(estimated_3d) ) - i  )
        sliding_window_start = i - sliding_window_length
        sliding_window_end = i + sliding_window_length +1

        average_skeleton_gravity = np.mean( np.array(estimated_3d [sliding_window_start: sliding_window_end ] ), axis=0 )
        average_gravity = np.mean( average_skeleton_gravity, axis=0 )
        smoothing_offset = current_gravity - average_gravity

        joints_3d = joints_3d + smoothing_offset.reshape(1, 3)
        smoothed_3d_joints = np.vstack([smoothed_3d_joints, joints_3d[None, :, :]])

    return smoothed_3d_joints

def get_2d_joints(kp_dict, sess, view, tracklet_id):
    traj = kp_dict[sess][view][tracklet_id]
    joints_2d = np.array( [ _traj["keypoints"][:, :2] for _traj in traj ] )
    return joints_2d

def save_3d_joints_estimation():
    """Save reconstructed 3D joints to pickle."""
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

def get_2_view_mpjpe(data, view1, view2, P1, P2, sess, tracker_id):
    joints_3d = estimate_3D_points(view_nums=[view1, view2], sess_num=sess, tracker_id=tracker_id, smoothing=True)
    joints_2d_view1= get_2d_joints( data, sess, view1, tracker_id )
    joints_2d_view2 = get_2d_joints( data, sess, view2, tracker_id )
    mpjpe1 = get_MPJPE(joints_3d, joints_2d_view1, P1)
    mpjpe2 = get_MPJPE(joints_3d, joints_2d_view2, P2)
    # print( f"label {track_id} view 1: {mpjpe1*2.54} cm. view 2: {mpjpe2*2.54} cm." )
    return mpjpe1, mpjpe2


if __name__ == "__main__":
    f = open('0.pickle', 'rb')
    data = pickle.load(f)
    K, k, H, traj_start_frames, start_frame, trajectory, extrinsic = get_param(view_num = 1,
                                                                               video_num = 0,
                                                                            tracker_id = 0)
    P1 = K@extrinsic

    K, k, H, traj_start_frames, start_frame, trajectory, extrinsic = get_param(view_num = 2,
                                                                               video_num = 0,
                                                                            tracker_id = 0)
    P2 = K@extrinsic

    K, k, H, traj_start_frames, start_frame, trajectory, extrinsic = get_param(view_num = 3,
                                                                               video_num = 0,
                                                                            tracker_id = 0)
    P3 = K@extrinsic
    P = [-1, P1, P2, P3]
    views = [1, 2, 3]
    for i in range(18):
        visualize_3D_joint_traj(views, sess_num=0, tracker_id=i, sample_rate=30, max_frame=10, smoothing=True)
        mpjpe1, mpjpe2 =  get_2_view_mpjpe(data, views[0], views[1], P[views[0]], P[views[1]], 0, tracker_id = i)
        _, mpjpe3 =  get_2_view_mpjpe(data, views[0], views[2], P[views[0]], P[views[2]], 0, tracker_id = i)
        print( f"label {i} view 1: {mpjpe1*2.54} cm. view 2: {mpjpe2*2.54} cm. view 3: {mpjpe3*2.54} cm." )
    
    # joints_3d = estimate_3D_points(view_nums=[2, 3], sess_num=0, tracker_id=1, smoothing=True)
    # exit()
    # visualize_3D_joint_traj( [1, 3], sess_num=0, tracker_id=0, sample_rate=30, max_frame=300)
    # joints_3d = estimate_3D_points(view_nums=[1, 2], sess_num=0, tracker_id=0, smoothing=False)


    # make_video_from_dir("./joint_3d_visualize/", "./joint_3d_visualize/3d_video.avi", fps=3)

    # Test MPJPE


    # mpjpe_list = []
    # for i in range( len(data[0][1])):
    #     mpjpe1, mpjpe2 =  get_2_view_mpjpe(data, 1, 2, P1, P2, 0, tracker_id = i)
    #     mpjpe_list.append( (mpjpe1+mpjpe2)/2 )
    #     print( f"label {i} view 1: {mpjpe1*2.54} cm. view 2: {mpjpe2*2.54} cm." )

    # print( np.mean( np.array(mpjpe_list) )*2.54 )