import scipy.io as sio
import numpy as np
import cv2
import os

COLOR_LIST = [(60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
              (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), 
              (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
              (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2

def to_homo(points):
    """Conver points to homogenous point.
    Args:
        points: Nx2 np.adarray points
    
    Returns:
        homo_points: Nx3 points    
    """
    if points.shape[1] == 3:
        return points
    return np.hstack( [points, np.ones( (points.shape[0], 1) )] )

def load_mat_dict(mat_path):
    """Load .mat fomat file."""
    return sio.loadmat(mat_path)

def video_to_frame(video_path, output_path, start_frame=0):
    """Read video and write frams to output_path."""
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count >= start_frame:
            write_path = os.path.join(output_path, f"frame_{str(count).zfill(4)}.png" )
            cv2.imwrite(write_path, image)     
            success,image = vidcap.read()
            if count % 600 == 0:
                print(f"finished frame {count}")
        count += 1

def make_videos(frames, output_path, fps=60):
    """Write a video given list of frames.

    Args:
        frames: list
        output_path: path to write the video
        fps: fps to write the video
    """
    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, frames[0].shape[:2][::-1])
    for f in frames:
        out.write(f)
    out.release()

def make_video_from_dir(dir_path, output_path, fps=60):
    """Write a video given list of frames.

    Args:
        dir_path: path to directory
        output_path: path to write the video
        fps: fps to write the video
    """
    frames = sorted(os.listdir(dir_path))
    # print(frames)
    frames = [ cv2.imread(os.path.join(dir_path, f_path)) for f_path in frames if ".png" in f_path ]
    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, frames[0].shape[:2][::-1])
    for f in frames:
        out.write(f)
    out.release()

def get_param( view_num, video_num, tracker_id ):
    """Get intrinsic matrix, homography, tracklet's starting frame, video's starting frame, tracklet's trajectories.
    
    Args:
        view_num: 1, 2, or 3 which camera's view.
        video_num: 0-7, video clip.
        tracker_id: tracklet's label. -1 for all label, list [1, 2, 3] for label 1, 2 and 3, or just integer for 1 label.
    
    Returns:
        K: Intrinsic matrix for the camera view_num.
        k: Distortion from lens.
        H: Homography to map from z=0 plane to camera's image plane.
        traj_start_frames: First frame to start tracking the tracklet after synchonize.
        start_frame: The fisrt frame for the video, used to synchronize the frames.
        trajectory: 3D trajectory on z=0 plane.
        extrinsic: extrinsic matrix from camara. view 1, 2, 3 -> left, right, center
    """
    if view_num == 0:
        raise ValueError("view num is 1-index, got 0")

    K = load_mat_dict(f"./video_data_n{view_num}/C0.mat")["K"]
    k = load_mat_dict(f"./video_data_n{view_num}/C0.mat")["k"]
    H = load_mat_dict(f"./video_data_n{view_num}/H0.mat")["H"]
    start_frame_txt_path = f"./video_data_n{view_num}/start_frames.txt"
    with open(start_frame_txt_path) as f:
        start_frame_lines = [line for line in f.readlines()]
    start_frame = int(start_frame_lines[video_num][2:]) # string manipulation.

    traj_start_frames = load_mat_dict(f"./label/{video_num}.mat")
    traj_start_frames = [s[0][0][0] for s in traj_start_frames["traj_starts"].T]

    trajectory = load_mat_dict(f"./label/{video_num}.mat")["trajectories"][0]
    if type(tracker_id) is list:
        traj_start_frames = [ traj_start_frames[_id] for _id in tracker_id ]
        trajectory = trajectory[ tracker_id ]

    elif tracker_id >= 0:
        traj_start_frames = traj_start_frames[tracker_id]
        trajectory = trajectory[tracker_id]

    if video_num in {0, 1, 2, 3}:
        extrinsic = load_mat_dict("./extrinsics/extrinsics_sess_0_1_2_3.mat")
    else:
        extrinsic = load_mat_dict("./extrinsics/extrinsics_sess_4_5_6_7.mat")

    # view 1, 2, 3 -> left, right, center
    # Use a dummy input at index 0 since view_num is 1-indexed
    R = [-1, extrinsic["R_left"], extrinsic["R_right"], extrinsic["R_center"]]
    t = [-1, extrinsic["t_left"], extrinsic["t_right"], extrinsic["t_center"]]
    extrinsic = np.hstack( [ R[view_num], t[view_num] ] )

    return K, k, H, traj_start_frames, start_frame, trajectory, extrinsic

def project_trajectories(trajectories, H):
    """Project trajectories on z=0 plane to camera frame given H.

    Args:
        trajectories: Nx3, 3d coordinate [x, y, 0].
        H: Homography.

    Returns:
        trajectories: Nx2, 2d [x, y] coordinate on new plane.
    """
    trajectories = np.hstack( [trajectories[:, :2], np.ones( (trajectories.shape[0], 1) )])
    trajectories = H.dot( trajectories.T ).T
    trajectories = trajectories/trajectories[:, -1, None]
    return trajectories[:, :2].astype(int)


def read_frame(video_dir, frame_path):
    """Tracking trajectory can be obtained with 2 camera view, but we have 3 cameras. So for some camera, it is possible to have trajectory
    without a valid frame. This function checks if the frame exist, is not, read the last exist frame.

    Args:
        video_dir: Directory of frames.
        frame_path: Path to read.

    Returns:
        np.array for read frame.    
    """
    if not os.path.exists(frame_path):
        frame_path = os.path.join( video_dir, sorted(os.listdir(video_dir))[-1] )
    return cv2.imread(frame_path)

def draw_trajtory( video_dir, output_dir, view_num, video_num, tracker_id):
    """Draw trajectory of each tracklet and write video.
    
    Args:
        video_dir: Path to the images from video.
        output_dir: Path to write intermidiate frames.
        view_num: 1, 2, 3, corresponding to left, right, center cameras
        video_num: Session num. 0 to 7.
        tracker_id: Label for person(s) to track. 
                    List[int] to track all the labels in the list. -1 to track every label.
    
    Returns:
        None.
    """
    video_dir =  video_dir.format( view_num, video_num )
    output_dir = output_dir.format( view_num )

    print(f"drawing {view_num}, video_num {video_num}, label {tracker_id}")
    if type(tracker_id) is list or tracker_id == -1:
        multi_track = True
    else:
        multi_track = False

    K, k, H, tracklet_start_frame, video_start_frame, trajectories, _ = get_param(view_num = view_num, 
                                                                            video_num = video_num, 
                                                                            tracker_id = tracker_id)
    
    if multi_track:
        num_of_frames = [len(_traj) for _traj in trajectories]
        last_frame_num = max( [_label_s + _nof + video_start_frame - 1 for _label_s, _nof in zip(tracklet_start_frame, num_of_frames)] ) 
        trajectories = [project_trajectories(_traj, H) for _traj in trajectories] 
    else:
        num_of_frames = len(trajectories)
        last_frame_num = tracklet_start_frame + num_of_frames + video_start_frame - 1
        trajectories = project_trajectories(trajectories, H)

    # due to sync, some view might not contain frame corresponds to last few trajectries.
    last_frame_path = os.path.join(video_dir, f"frame_{str(last_frame_num).zfill(4)}.png")
    last_frame = read_frame(video_dir, last_frame_path)

    if write_intermediate:
        all_frames = []
        if multi_track:
            first_frame_num = min(tracklet_start_frame)
            # tracklet_end_frame -> include
            tracklet_end_frame = [ _start + _nof - 1 for _start, _nof in zip(tracklet_start_frame, num_of_frames) ]
            last_frame_num = max(tracklet_end_frame)


            # Walk through frames
            # frame num in tracklet space
            # video_frame_num = video_start_frame + tracklet_space_frame_num
            for i in range( first_frame_num, last_frame_num+1 ):

                current_frame_num = video_start_frame + i
                current_frame_path = os.path.join( video_dir,  f"frame_{str(current_frame_num).zfill(4)}.png"  )
                current_frame = read_frame(video_dir, current_frame_path) # use read frame to prevent exceed time frame

                if (i-first_frame_num)%100 == 0:
                    print(f"Current process: frame num: {current_frame_num}, {i-first_frame_num}/{last_frame_num-first_frame_num}")

                # Walk through each tracklet
                for _label_id, (_start, _end) in enumerate(zip(tracklet_start_frame, tracklet_end_frame)):
                    if  _start <= i and _end >= i:
                        image_coor = trajectories[_label_id][current_frame_num - _start - video_start_frame ]
                        cv2.circle(current_frame, image_coor, 5, COLOR_LIST[_label_id%len(COLOR_LIST)], -1)

                imwrite_path = os.path.join(output_dir, f"frame{current_frame_num}.png")
                all_frames.append(current_frame)
                cv2.putText(current_frame, f"{video_start_frame}/{current_frame_num}", bottomLeftCornerOfText, font, fontScale, 
                            fontColor, thickness, lineType)
                cv2.imwrite(imwrite_path, current_frame)

        else:
            for i, traj in enumerate(trajectories):
                image_coor = traj
                current_frame_num = i + tracklet_start_frame + video_start_frame
                current_frame_path = os.path.join( video_dir,  f"frame_{str(current_frame_num).zfill(4)}.png"  )
                current_frame = read_frame(video_dir, current_frame_path)

                cv2.circle(current_frame, image_coor, 2, (0, 255, 0), -1)
                all_frames.append(current_frame)
                imwrite_path = os.path.join(output_dir, f"frame{current_frame_num}.png")
                cv2.putText(current_frame, f"{video_start_frame}/{current_frame_num}", bottomLeftCornerOfText, font, fontScale, 
                            fontColor, thickness, lineType)
                cv2.imwrite(imwrite_path, current_frame)

        make_videos( all_frames, f"view_{view_num}_video_{video_num}_track_id_{tracker_id}.avi", fps = 60 )
        print(f"Write to video path: view_{view_num}_video_{video_num}_track_id_{tracker_id}.avi")

    for i, _traj in enumerate(trajectories):
        if multi_track:
            for _coor in _traj:
                cv2.circle(last_frame, _coor,  2, COLOR_LIST[i%len(COLOR_LIST)], -1)
        else:
            cv2.circle(last_frame, _traj,  2, (0, 255, 0), -1)

    cv2.imwrite(f"./whole_traj/last_frame_{view_num}_video_{video_num}_label_{tracker_id}.png", last_frame)

def to_skew(pts):
    """Transform 3d vector(s) to skew symmetric matrix."""
    x, y, z = pts.T
    x, y, z = x[:, None], y[:, None], z[:, None]
    zeros = np.zeros_like( x )
    skew = np.hstack( [zeros, -z, y,
                       z, zeros, -x,
                       -y, x, zeros ] )
    return skew.reshape(-1, 3, 3)

def triangulate(P_list, pts_list):
    """Triangulation to get 3D coordinates.

    Args:
        p1: 3x4 camera parameter
        p2: 3x4 camera parameter
        pts1: Nx3 points
        pts2: Nx3 points

    Returns:
        _3d_points: Nx4 points in 3D    
    """
    if len(P_list) != len(pts_list):
        raise ValueError(f"Expect same number of views and corresponding keypoints. Got {len(P_list)} and {len(pts_list)}.")

    skew_list = [ to_skew(to_homo(pts)) for pts in pts_list ]
    cons_list = [ pts_skew@P for pts_skew, P in zip(skew_list, P_list) ]

    _3d_points = np.empty( (0, 3) )
    for _cons in zip(*(cons_list)):
        A = np.vstack( [ single_view_cons[:2] for single_view_cons in _cons] )
        _, _, VT = np.linalg.svd(A)
        p3d = VT[-1]
        p3d = (p3d/p3d[-1])[:3]
        _3d_points = np.vstack([_3d_points, p3d] )

    return _3d_points

if __name__ == "__main__":
    # video_to_frame("./video_data_n1/0.mp4", "./video_data_n1/tepper_0_frames", start_frame=1294)

    view_num = 2
    video_num = 0
    tracker_id = 4
    write_intermediate = True
    write_intermediate = True
    video_dir =  "./video_data_n{}/tepper_{}_frames"
    output_dir = "./test_output_n{}"

    # example of 3 ways to use draw_trajextory, list of tracklets, a tracklet and all tracklet
    # draw_trajtory( video_dir, output_dir, view_num, video_num, [0, 1, 2, 3])
    # draw_trajtory( video_dir, output_dir, view_num, video_num, 0)
    # draw_trajtory( video_dir, output_dir, view_num, video_num, -1)

    extrinsic = load_mat_dict("./extrinsics_sess_0_1_2_3.mat")
    # view 1, 2, 3 -> left, right, center
    R1, R2, R3 = extrinsic["R_left"], extrinsic["R_right"], extrinsic["R_center"]
    t1, t2, t3 = extrinsic["t_left"], extrinsic["t_right"], extrinsic["t_center"]