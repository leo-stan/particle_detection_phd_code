# Package to detect airborne particles in a LIDAR pointcloud


## Data preparation

- bag_formatter: Select parts of data in 3D space and put a label on it
- extract_rosbags: Take in formatted bags and split in training/testing/validation sets


### Steps
1. Select bags of data in bag_formatter.h
2. run bag_formatter.cpp
3. Select formatted bags in extract_rosbags.py
4. Run rosbag.py

## Topic prediction

'''bash
$rosparam set use_sim_time true
$roslaunch smoke_detection transforms.launch
$rosrun smoke_detection scan_formatter
$rosbag play whatever bag you want to predict
'''


 ## Traditional ML approach (Random Forest)

- LidarDatasetHC.py
- train_model.py
- topic_prediction.py

### Steps

1. Prepare data
2. Run train_model.py
3. Run topic_prediction.py

 ## Deep Learning approach

- LidarDataset.py
- train_smokenet.py
- smokenet_topic_prediction.py


## Instructions for ETH labeled smoke dataset

### Data preparation:
1. Choose a bag file with /velodyne_points_dual topic (e.g. smoke.bag)
2. Run convert_rosbag_to_numpy.ipynb script on smoke.bag, it generates smoke.npy with [pointcloud,images]
3. Run 1_labeling_pipeline_dual.py from Julian on smoke.npy, it generates smoke_labeled.npy which adds a columns with labels to each point of lidar pcl
4. Run 2_labeling_pipeline_spaces.py on smoke_labeled.npy with the right planes to refine labeling and remove any mislabeled points (from moving objects most of the time), it generates smoke_labeled_spaces.npy
5. You can visualise labeled_points on Rviz using visualise_labeled_lidar_pcl.ipynb on smoke_labeled_spaces.npy to make sure the labeling is right
6. Run convert_julian_pcl.ipynb on smoke_labeled_spaces.npy to convert his label formatting to mine (one column with label number instead of a boolean column for each label), it generates smoke_labeled_spaces_converted.npy and smoke_imgs.npy
7. Run generate_lidar_voxels.py on smoke_labeled_spaces_converted.npy to transform raw lidar scans into a set of voxels already arranged for smokenet tensors (smoke voxels, non-smoke voxels, and scans_voxels), this will be used by my custom dataset to train smokenet

### Training of smokenet with ETH labeled data
8. Run train_smokenet_eth.py with whatever prepared dataset
9. Run evaluate_models.py to compare different trained models
10. run visualise_prediction.py to visualise the prediction in Rviz /smokenet_prediction_pcl, /smokenet_prediction_vox, velodyne_points_labeled

## TODO:

- Update requirements.txt
- Make the data preparation process a bit more user friendly
- Integrate FCN classifier


## New instructions

1. Run convert_rosbag_to_numpy.py
2. Either run Julian classification on pcl.npy to obtain pcl_labeled_spaces.npy or copy files from FSR paper
3. Run convert_julian_pcl_fsr.npy to obtain pcl_labeled_spaces_converted.npy
4. (Optional) You can visualise these in Rviz using visualise_eth_labeled_dataset.py
4. Run remove_empty_img_dataset.py to remove all black images and corresponding velodyne scans (this is from velodyne scans that don't have a corresponding multisense image in the rosbags)
5. Run project_lidar_pts_in_images.py to obtain pcl_labeled_spaces_converted_projected.npy and pcl_labeled_spaces_converted_cropped.npy
6. (Optional) You can now run visualise_pcl_projected_imgs.py to visualise lidar points in images
7. Run generate_image_labels.py to obtain images/ and image_labels/ as well as image_labels.npy


### LiDAR dataset

- dataset_name/
    - scan_pcls/
        - 0.npy
        - 1.npy
        - ...
        - M.npy
    - scan_voxels/
        - coords_0.npy
        - coords_1.npy
        - ...
        - coords_M.npy
        - labels_0.npy
        - labels_1.npy
        - ...
        - labels_M.npy
    config.yaml
    scaler.pkl

### Stereo dataset

- dataset_name/
    - scan_pcls/
        - 0.npy
        - 1.npy
        - ...
        - M.npy
    - scan_voxels/
        - coords_0.npy
        - coords_1.npy
        - ...
        - coords_M.npy
        - labels_0.npy
        - labels_1.npy
        - ...
        - labels_M.npy
config.yaml
scaler.pkl


## Multisense process

1. rosbag -> convert_rosbag_to_numpy.py -> data/dataset_name/multi_pcl.npy
2. multi_pcl.npy -> sample_multisense_pcl.py -> multi_pcl_sampled.npy
3. multi_pcl_sampled.npy -> labeling -> multi_pcl_sampled_labeled.npy
4. multi_pcl_sampled_labeled.npy -> convert_julian_pcl_fsr.py -> multi_pcl_sampled_labeled_converted.npy
5. multi_pcl_sampled_labeled_converted.npy -> project_lidar_pts_in_images.py -> (multi_pcl_sampled_labeled_converted_cropped.npy,multi_pcl_sampled_labeled_converted_projected.npy) 
