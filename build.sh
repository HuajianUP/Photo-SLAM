# DBoW2
cd ./ORB-SLAM3/Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release # add OpenCV_DIR definitions if needed, example:
#cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/home/rapidlab/libs/opencv/lib/cmake/opencv4
make -j

cd ../../g2o

# g2o
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../Sophus

# Sophus
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# ORB_SLAM3
cd ../../../Vocabulary
echo "Uncompress vocabulary ..."
tar -xf ORBvoc.txt.tar.gz
cd ..

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release # add OpenCV_DIR definitions if needed, example:
#cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/home/rapidlab/libs/opencv/lib/cmake/opencv4
make -j8

# Photo-SLAM
echo "Building Photo-SLAM ..."
cd ../..
mkdir build
cd build
cmake .. # add Torch_DIR and/or OpenCV_DIR definitions if needed, example:
#cmake .. -DTorch_DIR=/home/rapidlab/libs/libtorch/share/cmake/Torch -DOpenCV_DIR=/home/rapidlab/libs/opencv/lib/cmake/opencv4
make -j8
