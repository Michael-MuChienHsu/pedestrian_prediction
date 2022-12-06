import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

def get_cross_matrix(vec):
    mat = [[0,-vec[2],vec[1]],[vec[2],0,-vec[0]],[-vec[1],vec[0],0]]
    mat = np.array(mat)
    return mat

def my_smooth_3d_joints(estimated_3d, max_sliding_window_length, use_high_conf_filter, high_conf_mask, iteration):
    '''
    estimated_3d: N x 17 x 3  
    '''
    estimated_3d = np.stack(estimated_3d)
    head_idx = np.array([0,1,2,3,4])
    shoulder_idx = np.array([5,6])
    elbow_idx = np.array([7,8])
    hand_idx = np.array([9,10])
    hip_idx = np.array([11,12])
    knee_idx = np.array([13,14])
    feet_idx = np.array([15,16])
   
    smoothed_3d_joints = estimated_3d.copy()

    pair_name = [
            'right shoulder - elbow',
            'right elbow - hand',
            'left shoulder - elbow',
            'left elbow - hand',
            'right hip - knee',
            'right knee - foot',
            'left hip - knee',
            'left knee - foot',
            ]
    joint_pairs = [
            # right
            [6,8],    # shoulder - elbow 
            [8,10],   # elbow - hand
            # left
            [5,7],    # shoulder - elbow
            [7,9],    # elbow - hand
            # right
            [12,14],  # hip - knee
            [14,16],  # knee - foot
            # left
            [11,13],  # hip - knee
            [13,15]   # knee - foot
            ]

    for e,(name,pair) in enumerate(zip(pair_name,joint_pairs)):
        #if e < 4:
        #    continue
        print(name)
        anchor = estimated_3d[:, pair[0], :]
        joint = estimated_3d[:, pair[1], :]
        smoothed_joint = smooth_angle(anchor,joint,max_sliding_window_length,"%s_%d"%(name, iteration))
        smoothed_3d_joints[:,pair[1],:] = smoothed_joint

    return smoothed_3d_joints

def smooth_angle(anchor, joint, max_sliding_window_length, pair_name):
    '''
    anchor: N x 3
    joint: N x 3
    '''
    N = anchor.shape[0] 
    # to local coordinate system
    jointn = joint - anchor
    jointn_norm = np.linalg.norm(jointn, axis=1)
    mean_length = jointn_norm.mean()
    direction_vec = jointn / jointn_norm[:,np.newaxis] 

    #zaxis = np.array([0,0,1])
    zaxis = direction_vec[0,:]
    # rotation matrix from zaxis to vec
    angles = []
    for i in range(N):
        vec = direction_vec[i]
        v = np.cross(zaxis, vec)
        s = np.linalg.norm(v)
        c = np.dot(zaxis, vec)
        vcross = get_cross_matrix(v)
        rotmat = np.eye(3) + vcross + vcross @ vcross /(1+c)
        
        r = R.from_matrix(rotmat).as_euler('xyz', degrees=True)
        angles.append(r)
    
    angles = np.stack(angles)
    # smooth angle
    smoothed_angles = []
    for i in range(N):
        start = max(0, i-max_sliding_window_length)
        end = min(N-1, i+max_sliding_window_length)
        angle = angles[start:end+1,:].mean(axis=0)
        smoothed_angles.append(angle)
    
    smoothed_angles = np.stack(smoothed_angles)

    smoothed_direction_vec = []
    for i in range(N):
        rotmat = R.from_euler('xyz',smoothed_angles[i],degrees=True).as_matrix()
        vec = rotmat @ zaxis
        smoothed_direction_vec.append(vec)
    smoothed_direction_vec = np.stack(smoothed_direction_vec)
    smoothed_jointn = smoothed_direction_vec * mean_length
   
    smoothed_joint = smoothed_jointn + anchor

    plotx = np.arange(N)
    fig = plt.figure()
    plt.plot(plotx, jointn_norm,label='length')
    plt.plot(plotx, angles[:,0],label='angle x')
    plt.plot(plotx, angles[:,1],label='angle y')
    plt.plot(plotx, angles[:,2],label='angle z')
    
    plt.plot(plotx, smoothed_angles[:,0],label='smoothed angle x')
    plt.plot(plotx, smoothed_angles[:,1],label='smoothed angle y')
    plt.plot(plotx, smoothed_angles[:,2],label='smoothed angle z')
    plt.title(pair_name)

    plt.legend()
    plt.savefig('figure/%s.jpg'%pair_name)
    #plt.show()
     

    return smoothed_joint

