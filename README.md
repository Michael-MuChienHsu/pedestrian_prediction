# Modeling and Understanding Pedestrian Behavior

## Contents
0. Data setup.
1. 2D joint estimation with Byte Track.
2. Triangulation using multiview 2D joints.
3. 3D Trajectrory prediction network.
4. References

## Data setup
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
```

## 2D joint estimation with Byte Track.

##  Triangulation using multiview 2D joints.
`python triangulation.py -y "triangulation.yaml"`

## 3D Trajectrory prediction network.

## References
1.  [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](https://arxiv.org/abs/2110.06864) , Yifu Z. et al. ECCV 2022. [Github](https://github.com/ifzhang/ByteTrack)
