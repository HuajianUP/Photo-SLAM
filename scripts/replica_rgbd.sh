#!/bin/bash

for i in 0 1 2 3 4
do
../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/rapidlab/dataset/VSLAM/Replica/office0 \
    ../results/replica_rgbd_$i/office0 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office1.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/rapidlab/dataset/VSLAM/Replica/office1 \
    ../results/replica_rgbd_$i/office1 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office2.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/rapidlab/dataset/VSLAM/Replica/office2 \
    ../results/replica_rgbd_$i/office2 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office3.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/rapidlab/dataset/VSLAM/Replica/office3 \
    ../results/replica_rgbd_$i/office3 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office4.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/rapidlab/dataset/VSLAM/Replica/office4 \
    ../results/replica_rgbd_$i/office4 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/room0.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/rapidlab/dataset/VSLAM/Replica/room0 \
    ../results/replica_rgbd_$i/room0 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/room1.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/rapidlab/dataset/VSLAM/Replica/room1 \
    ../results/replica_rgbd_$i/room1 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/room2.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/rapidlab/dataset/VSLAM/Replica/room2 \
    ../results/replica_rgbd_$i/room2 \
    no_viewer
done