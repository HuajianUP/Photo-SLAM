#!/bin/bash

dates=("2023-08-18" "2023-09-15" "2024-01-13" "2024-04-11" "2024-05-29_1" "2024-05-29_2" "2024-05-29_3" "2024-05-29_4")

for date in "${dates[@]}"; do
    echo "Processing date: $date"

    trials=("0" "1" "2" "3" "4")

    for trial in "${trials[@]}"; do
        echo "  Trial: $trial"

        ../bin/stihl_mono \
            ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
            ../cfg/ORB_SLAM3/Monocular/Stihl/d435i.yaml \
            ../cfg/gaussian_mapper/Monocular/Stihl/stihl_mono.yaml \
            /workspace/mounted_directory/media/fabian/data_recording_r/kwald/drosselweg/flaeche1/$date/tum/d435i \
            ../results/short/mono/$date/$trial \
            no_viewer

        ../bin/stihl_rgbd \
            ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
            ../cfg/ORB_SLAM3/RGB-D/Stihl/d435i.yaml \
            ../cfg/gaussian_mapper/RGB-D/Stihl/stihl_rgbd.yaml \
            /workspace/mounted_directory/media/fabian/data_recording_r/kwald/drosselweg/flaeche1/$date/tum/d435i \
            ../cfg/ORB_SLAM3/RGB-D/Stihl/associations/$date.txt \
            ../results/short/rgbd/$date/$trial \
            no_viewer
    done
done

echo "All dates processed."