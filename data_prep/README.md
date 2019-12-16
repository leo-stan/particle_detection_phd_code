# Description
This folder contains files to pre-process ROS bag files into formatted datasets

# Data preparation steps
1. convert_rosbag_to_numpy.py
2. Julian labeling of LiDAR
3. sync_dataset.py
4. sample_multisense_pcl.py
5. label_pcl.py
6. project_pcls.py
7. refine_labelling.py
8. project_multi_in_imgs.py
9. generate_image_labels.py

# Visualisation

- visualise_dataset.py: to visualise a dataset using Rviz

# Input
A rosbag with velodyne and multisense data

# Output
shown in example.png

# Dataset preparation

### General dataset

- scene-name/
    - image_labels/
    - images/
    - image_labels.npy (list of image labels) [N,RGB]
    - images.npy (list of image) [N,RGB]
    - multi_pcl.npy (List of Multisense 3D points for each scan [N,XYZRGB])
    - multi_pcl_sampled.npy (A sampled version of multi_pcl.npy to make computation possible)
    - multi_pcl_sampled_projected.npy (3D points projected in image [[N,XYRGB]])
    - pcl.npy (List of LiDAR 3D points for each scan [N,XYZRingIntensity])
    - pcl_labeled_spaces_converted.npy (List of Labeled LiDAR 3D points for each scan converted into my multi-echo format from Julian's labeling [N,XYZIntensityEchoLabel])
    - pcl_labeled_spaces_converted_cropped.npy (Cropped for Mulitsense FoV)
    - pcl_labeled_spaces_converted_projected.npy (Projected into Multisense Image) [N,XYIntensityEchoLabel]
    