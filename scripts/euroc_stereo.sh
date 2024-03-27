#!/bin/bash

for i in 0 1 2 3 4
do
../bin/euroc_stereo \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC.yaml \
    ../cfg/gaussian_mapper/Stereo/EuRoC/EuRoC.yaml \
    /home/rapidlab/dataset/VSLAM/EuRoC/MH_01_easy \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC_TimeStamps/MH01.txt \
    ../results/euroc_stereo_$i/MH_01_easy \
    no_viewer

../bin/euroc_stereo \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC.yaml \
    ../cfg/gaussian_mapper/Stereo/EuRoC/EuRoC.yaml \
    /home/rapidlab/dataset/VSLAM/EuRoC/MH_02_easy \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC_TimeStamps/MH02.txt \
    ../results/euroc_stereo_$i/MH_02_easy \
    no_viewer

../bin/euroc_stereo \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC.yaml \
    ../cfg/gaussian_mapper/Stereo/EuRoC/EuRoC.yaml \
    /home/rapidlab/dataset/VSLAM/EuRoC/V1_01_easy \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC_TimeStamps/V101.txt \
    ../results/euroc_stereo_$i/V1_01_easy \
    no_viewer

../bin/euroc_stereo \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC.yaml \
    ../cfg/gaussian_mapper/Stereo/EuRoC/EuRoC.yaml \
    /home/rapidlab/dataset/VSLAM/EuRoC/V2_01_easy \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC_TimeStamps/V201.txt \
    ../results/euroc_stereo_$i/V2_01_easy \
    no_viewer
done