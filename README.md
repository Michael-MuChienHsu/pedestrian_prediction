# Modeling and Understanding Pedestrian Behavior

## Contents
1. [Data setup](#data_setup).<br />
2. [Data2D joint estimation with Byte Track](#use_bytetrack).<br />
3. [Triangulation using multiview 2D joints](#triangulation).<br />
4. [3D Trajectrory prediction network](#network).<br />
5. [References](#ref).<br />

##  <a name="data_setup"></a>Data setup
```
3d_extracted_joints
    └——————0.pickle
    └——————1.pickle
    └——————...
    └——————7.pickle
labels
    └——————0_frame.txt
    └——————0.mat
    └——————...
    └——————7_frame.txt
    └——————7.txt
joint_3d_visualize
utils.py
triangulation.yaml
triangulation_utils.py
triangulation.py
```

## <a name="use_bytetrack"></a>2D joint estimation with Byte Track.

## <a name="triangulation"></a>Triangulation using multiview 2D joints.
`python triangulation.py -y "triangulation.yaml"`

## <a name="network"></a>3D Trajectrory prediction network.

## <a name="ref">References
1. [**Towards Rich, Portable, and Large-Scale Pedestrian Data Collection**](https://arxiv.org/abs/2203.01974) Wang, A., Biswas, A., Admoni, H., & Steinfeld, A. (2022) [Project website](https://tbd.ri.cmu.edu/resources/tbd-social-navigation-datasets/)
2. [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](https://arxiv.org/abs/2110.06864) , Yifu Z. et al. ECCV 2022. [Github](https://github.com/ifzhang/ByteTrack)
 
