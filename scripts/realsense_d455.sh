#!/bin/bash

../bin/realsense_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/RealCamera/realsense_d455_rgbd.yaml \
    ../cfg/gaussian_mapper/RGB-D/RealCamera/realsense_rgbd.yaml \
    ../results/realsense_d455_rgbd