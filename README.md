# Modeling and Understanding Pedestrian Behavior
In this project we proposed a new method to predict pedestrain's trajectory in 3D space with multuple 2D views. Our methoed combines deep learning based method to tackle the difficult joint detection tasks and trajextory prediction task, and geometry methods to estimate joint poses from different views. <br />
This project focuses on complex predestrain walking scenes recorded with bird-eye-view cameras.

## Contents
1. [Data setup](#data_setup).<br />
2. [Data2D joint estimation with Byte Track](#use_bytetrack).<br />
3. [Triangulation using multiview 2D joints](#triangulation).<br />
4. [3D Trajectrory prediction network](#network).<br />
5. [References](#ref).<br />

##  <a name="data_setup"></a>Data setup
There are 8 videos in the [TBD dataset](https://arxiv.org/abs/2203.01974).

***Folders***: <br />
***3d_extracted_joints***: joints extracted from Byte track.<br />
***video_data_n1***, ***video_data_n2***, ***video_data_n3***: Files C_.mat: intrinsic matrix, H_.mat: homography. for each video.<br />
***label***: Tracklets trajectories.  <br />
***extrinsics***: Extrinsic matrix. Video (session) 0 1 2 3 share same extrinsic and 4 5 6 7 share same extrinsics.  <br />

We only provide 0.pickle for 3d joints in this repo, others can be downloaded at [google drive link](https://drive.google.com/drive/folders/1tIlMNJRF0iSb5K90ICHJPa-NwZdKWWQc?usp=share_link)
```
3d_extracted_joints
    └——————0.pickle
    └——————...
video_data_n1/video_data_n2/video_data_n3
    └——————C_0.mat
    └——————H_0.mat
    └——————...
    └——————start_frames.txt
labels
    └——————0_frame.txt
    └——————0.mat
    └——————...
extrinsics
    └——————extrinsics_sess_0_1_2_3.mat
    └——————extrinsics_sess_4_5_6_7.mat    
joint_3d_visualize
utils.py
triangulation.yaml
triangulation_config.yaml
triangulation_utils.py
triangulation.py
```

## <a name="use_bytetrack"></a>2D joint estimation with Byte Track.

## <a name="triangulation"></a>Triangulation using multiview 2D joints.
To get 3d joints with triangulation: `python triangulation.py -y "./config/triangulation.yaml"` <br />
Thanks [wenwuX](https://github.com/wenyuX) for helping anti-jittering in 3D joint smoothing.

## <a name="network"></a>Baseline 3D Trajectrory prediction model.
Thanks for Allen providing his [previous work](http://www.cs.cmu.edu/~epxing/Class/10708-19/assets/project/final-reports/project19.pdf) as our baseline model. Our baseline model is built on top of his work. <br />
To train prediction model: `python train_model.py -y "./config/train_config.yaml"`

## <a name="ref">References
1. <a name="TBD_dataset">[**Towards Rich, Portable, and Large-Scale Pedestrian Data Collection**](https://arxiv.org/abs/2203.01974) Wang, A., Biswas, A., Admoni, H., & Steinfeld, A. (2022) [Project website](https://tbd.ri.cmu.edu/resources/tbd-social-navigation-datasets/)
2. [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](https://arxiv.org/abs/2110.06864) , Yifu Z. et al. ECCV 2022. [Github](https://github.com/ifzhang/ByteTrack)
3. <a name="allen_proj"></a>[**Pedestrian Trajectory Prediction with Graph Neural Networks**](http://www.cs.cmu.edu/~epxing/Class/10708-19/assets/project/final-reports/project19.pdf), Allan Wang, Zirui Wang , and Wentao Yuan.

 
---------------------------------------
## TODO:
use_high_conf_filter in `./triangulation_utils.py/visualize_3D_joint_traj` <br />
smooth z-only in `./triangulation_utils.py/visualize_3D_joint_traj` <br />
Modulize Visualize 2d joints in `./triangulation_utils.py/visualize_3D_joint_traj` <br />
