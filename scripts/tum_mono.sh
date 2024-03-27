#!/bin/bash

for i in 0 1 2 3 4
do
../bin/tum_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_desk.yaml \
    ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/rapidlab/dataset/VSLAM/TUM/rgbd_dataset_freiburg1_desk \
    ../results/tum_mono_$i/rgbd_dataset_freiburg1_desk \
    no_viewer

../bin/tum_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg2_xyz.yaml \
    ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/rapidlab/dataset/VSLAM/TUM/rgbd_dataset_freiburg2_xyz \
    ../results/tum_mono_$i/rgbd_dataset_freiburg2_xyz \
    no_viewer

../bin/tum_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_long_office_household.yaml \
    ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/rapidlab/dataset/VSLAM/TUM/rgbd_dataset_freiburg3_long_office_household \
    ../results/tum_mono_$i/rgbd_dataset_freiburg3_long_office_household \
    no_viewer
done