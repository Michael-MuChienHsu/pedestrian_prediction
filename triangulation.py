import pickle
from utils import get_param, read_frame, triangulate, make_videos
import cv2
import matplotlib.pyplot as plt
import numpy as np

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 1
lineType               = 2

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
    image_path =  f"./video_data_n{view_num}/tepper_{sess_num}_frames/frame_{str(frame_num).zfill(4)}.png"
    img = cv2.imread(image_path)
    min_x = min(kp[:, 0])
    min_y = min(kp[:, 1])
    text_x  = (kp[:, 0] - min_x)*4 + min_x 
    text_y  = (kp[:, 1] - min_y)*4 + min_y 
    
    for i, _x_y in enumerate(kp):
        cv2.circle( img, _x_y, 5, (0, 0, 255),  -1)
        
        cv2.putText(img, f"{i}", (text_x[i], text_y[i]), font, fontScale, 
                            fontColor, thickness, lineType)        
    cv2.imwrite(f"test_skeleton_{view_num}.png", img)

def visualize_3D_joint_traj(view_num_1, view_num_2, sess_num, tracker_id, sample_rate=20, max_frame = 10):
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
    """
    traj1 = data[sess_num][view_num_1][tracker_id]
    K1, k1, H1, traj_start_frames1, start_frame1, trajectory1, extrinsic1 = get_param(view_num = view_num_1,
                                                                                      video_num = sess_num,
                                                                                      tracker_id = tracker_id)
    P1 = K1@extrinsic1
    # DEBUG
    # show_2d_joint(view_num=view_num_1,
    #               sess_num=sess_num,
    #               frame_num=start_frame1+traj_start_frames1,
    #               kp=traj1[0]["keypoints"][:, :2].astype(int))
    
    traj2 = data[sess_num][view_num_2][tracker_id]
    K2, k2, H2, traj_start_frames2, start_frame2, _, extrinsic2 = get_param(view_num = view_num_2,
                                                                                      video_num = sess_num,
                                                                                      tracker_id = tracker_id)
    P2 = K2@extrinsic2

    kp1 = np.array([ traj["keypoints"][:, :2] for traj in traj1 ]).reshape((-1, 2))
    kp2 = np.array([ traj["keypoints"][:, :2] for traj in traj2 ]).reshape((-1, 2))
    estimated_3d = triangulate(P1, P2, kp1, kp2)
    estimated_3d = estimated_3d.reshape(-1, 17, 3)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    color_map = plt.get_cmap("jet")
    estimated_3d = [estimated_3d[i] for i in range(0, len(estimated_3d), sample_rate)][:max_frame]
    trajectory1 = [ trajectory1[i] for i in range(0, len(trajectory1), sample_rate) ]
    for i, (joints_3d) in enumerate(estimated_3d):
        z_min = min(joints_3d[:, 2])
        joints_3d = joints_3d.T
        
        # some trick to align ppl to z = 0 surface
        joints_3d = (joints_3d)*( np.array( [1, 1, -1] ).reshape((-1, 1)) )
        max_z = max(joints_3d[:][-1])
        min_z = min(joints_3d[:][-1])
        joints_3d[:][-1] = joints_3d[:][-1] - min_z
        # print(f"person height: {2.54*(max_z-min_z)} cm")
        print(min_z)

        start_3d, end_3d = get_skeleton_start_end_point(joints_3d)
        X, Y, Z = joints_3d

        color = color_map(i/float(len(estimated_3d)))
        ax.scatter(X, Y, Z, color = color )
        
        # ax.scatter(trajectory1[i][0], trajectory1[i][1], trajectory1[i][2], color = color )
        ax.scatter( *(trajectory1[i]), color = color )

        for _s, _e in zip( start_3d, end_3d ):
            ax.plot([_s[0], _e[0]], [_s[1], _e[1]], [_s[2], _e[2] ], color = color )

        plt.savefig(f"./joint_3d_visualize/track_frame_{i}.png")
    plt.savefig("./joint_3d_visualize/joints_3d.png")

    # DEBUG
    # show_2d_joint(view_num_1, sess_num, traj_start_frames1+start_frame1, kp1.astype(int))
    # show_2d_joint(view_num_2, sess_num, traj_start_frames2+start_frame2, kp2.astype(int))

    # reprojected1 = P1.dot( np.vstack( [X, Y, Z, np.ones_like(X)] ) ).T
    # reprojected1 = (reprojected1/reprojected1[:, -1, None])[:, :2]

    # reprojected2 = P2.dot( np.vstack( [X, Y, Z, np.ones_like(X)] ) ).T
    # reprojected2 = (reprojected2/reprojected2[:, -1, None])[:, :2]

    # for (true_x1, true_y1), (true_x2, true_y2), (_x1, _y1), (_x2, _y2) in zip( kp1, kp2, reprojected1, reprojected2 ):
    #     print(f"kp1 {true_x1, true_y1}, repro kp1 {_x1, _y1}, kp2 {true_x2, true_y2}, repro kp2 {_x2, _y2}")

def get_3D_point(view_num_1, view_num_2, sess_num, tracker_id, sample_rate=1):
    """Apply triangualtion on joints detected on 2 views to get 3D coordinate.
    Note that the Z-axis is upside down, we flipped the z-axis and force "stand on the ground constraint".

    Args:
        view_num_1: one view from static camera, 1-indexed.
        view_num_2: second view from static camera, 1-indexed.
        sess_num: video_num ranges from 0-7.
        tracker_id: label of whom tracked.
        sample_rate: the origin video is in 60 FPS, use sample rate to lower fps.

    Returns:
        estimated_3d: N x 17 x 3 joints.

    """
    traj1 = data[sess_num][view_num_1][tracker_id]
    K1, k1, H1, _, _, _, extrinsic1 = get_param(view_num = view_num_1,
                                                video_num = sess_num,
                                                tracker_id = tracker_id)
    P1 = K1@extrinsic1
    
    traj2 = data[sess_num][view_num_2][tracker_id]
    K2, k2, H2, _, _, _, extrinsic2 = get_param(view_num = view_num_2,
                                                video_num = sess_num,
                                                tracker_id = tracker_id)
    P2 = K2@extrinsic2

    kp1 = np.array([ traj["keypoints"][:, :2] for traj in traj1 ]).reshape((-1, 2))
    kp2 = np.array([ traj["keypoints"][:, :2] for traj in traj2 ]).reshape((-1, 2))
    estimated_3d = triangulate(P1, P2, kp1, kp2)
    estimated_3d = estimated_3d.reshape(-1, 17, 3)

    # Flip z axis.
    estimated_3d[:, :, -1] *= -1

    # Force "stand on floor constraint".
    min_z = np.min( estimated_3d[:, :, -1], axis=1 ).reshape(-1, 1)
    estimated_3d[:, :, -1] -= min_z

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

def save_3d_joints_estimation():
    """
    """
    pass

if __name__ == "__main__":
    f = open('0.pickle', 'rb')
    data = pickle.load(f)
    visualize_3D_joint_traj(view_num_1=1, view_num_2=2, sess_num=0, tracker_id=0)
    # joints_3d = get_3D_point(view_num_1=1, view_num_2=2, sess_num=0, tracker_id=0)
