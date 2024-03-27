# Photo-SLAM
### [Homepage](https://huajianup.github.io/research/Photo-SLAM/) | [Paper](https://arxiv.org/abs/2311.16728)

**Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras** <br>
[Huajian Huang](https://huajianup.github.io)<sup>1</sup>, [Longwei Li](https://github.com/liquorleaf)<sup>2</sup>, Hui Cheng<sup>2</sup>, and [Sai-Kit Yeung](https://saikit.org/)<sup>1</sup> <br>
The Hong Kong University of Science and Technology<sup>1</sup>, Sun Yat-Sen University<sup>2</sup> <br>
In Proceedings of Computer Vision and Pattern Recognition Conference (CVPR), 2024<br>
![image](https://huajianup.github.io/thumbnails/Photo-SLAM_v2.gif "photo-slam")


## Prerequisites
```
sudo apt install libeigen3-dev libboost-all-dev libjsoncpp-dev libopengl-dev mesa-utils libglfw3-dev libglm-dev
```

<table>
    <thead>
        <tr>
            <th>Dependencies</th>
            <th colspan=3>Tested with</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>OS</td>
            <td>Ubuntu 20.04 LTS</td>
            <td>Ubuntu 22.04 LTS</td>
            <td>Jetpack 5.1.2</td>
        </tr>
        <tr>
            <td>gcc</td>
            <td>10.5.0</td>
            <td>11.4.0</td>
            <td>9.4.0</td>
        </tr>
        <tr>
            <td>cmake</td>
            <td>3.27.5</td>
            <td>3.22.1</td>
            <td>3.26.4</td>
        </tr>
        <tr>
            <td><a href="https://developer.nvidia.com/cuda-toolkit-archive">CUDA</a> </td>
            <td>11.8</td>
            <td>11.8</td>
            <td>11.4</td>
        </tr>
        <tr>
            <td><a href="https://developer.nvidia.com/rdp/cudnn-archive">cuDNN</a> </td>
            <td>8.9.3</td>
            <td>8.7.0</td>
            <td>8.6.0</td>
        </tr>
        <tr>
            <td><a href="https://opencv.org/releases">OpenCV</a> built with opencv_contrib and CUDA</td>
            <td>4.7.0</td>
            <td>4.8.0</td>
            <td>4.7.0</td>
        </tr>
        <tr>
            <td><a href="https://pytorch.org/get-started/locally">LibTorch</a> </td>
            <td colspan=2>cxx11-abi-shared-with-deps-2.0.1+cu118</td>
            <td>2.0.0+nv23.05-cp38-linux_aarch64</td>
        </tr>
        <tr>
            <td colspan=4>(optional) <a href="https://github.com/IntelRealSense/librealsense">Intel® RealSense™ SDK 2.0</a> </td>
        </tr>
        <tr>
            <td colspan=4>(Remark) Jetson AGX Orin Developer Kit we used is with 64GB and its power model was set to MAXN.</td>
        </tr>
    </tbody>
</table>

### Using LibTorch
If you do not have the LibTorch installed in the system search paths for CMake, you need to add additional options to `build.sh` help CMake find LibTorch. See `build.sh` for details. Otherwise, you can also add one line before `find_package(Torch REQUIRED)` of `CMakeLists.txt`:

[Option 1] Conda. If you are using Conda to manage your python packages and have installed compatible Pytorch, you could set the 
```
# [For Jatson Orin] To install Pytorch in Jatson developer kit, you can run the below commands
# export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
# pip install --no-cache $TORCH_INSTALL

set(Torch_DIR /the_path_to_conda/python3.x/site-packages/torch/share/cmake/Torch)
```

[Option 2] You cloud download the libtorch, e.g., [cu118](https://download.pytorch.org/libtorch/cu118) and then extract them to the folder `./the_path_to_where_you_extracted_LibTorch`. 
```
set(Torch_DIR /the_path_to_where_you_extracted_LibTorch/libtorch/share/cmake/Torch)
```

### Using OpenCV with opencv_contrib and CUDA
Take version 4.7.0 for example, look into [OpenCV realeases](https://github.com/opencv/opencv/releases) and [opencv_contrib](https://github.com/opencv/opencv_contrib/tags), you will find [OpenCV 4.7.0](https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz) and the corresponding [opencv_contrib 4.7.0](https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.tar.gz), download them to the same directory (for example, `~/opencv`) and extract them. Then open a terminal and run:
```
cd ~/opencv
cd opencv-4.7.0/
mkdir build
cd build

# The build options we used in our tests:
cmake -DCMAKE_BUILD_TYPE=RELEASE -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON -DWITH_NVCUVID=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 -DOPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-4.7.0/modules" -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_JASPER=ON -DBUILD_CCALIB=ON -DBUILD_JPEG=ON -DWITH_FFMPEG=ON ..

# Take a moment to check the cmake output, see if there are any packages needed by OpenCV but not installed on your device

make -j8
# NOTE: We found that compilation of OpenCV may stuck at 99%, this may be caused by the final linking process. We just waited it for a while until it completed and exited without errors.
```
To install OpenCV into the system path:
```
sudo make install
```
If you prefer installing OpenCV to a custom path by adding `-DCMAKE_INSTALL_PREFIX=/your_preferred_path` option to the `cmake` command, remember to help Photo-SLAM find OpenCV by adding additional cmake options. See `build.sh` for details. Otherwise, you can also add the following line to `CMakeLists.txt`, `ORB-SLAM3/CMakeLists.txt` and `ORB-SLAM3/Thirdparty/DBoW2/CMakeLists.txt`, just like what we did for LibTorch.
```
set(OpenCV_DIR /your_preferred_path/lib/cmake/opencv4)
```

## Installation of Photo-SLAM
```
git clone https://github.com/HuajianUP/Photo-SLAM.git
cd Photo-SLAM/
chmod +x ./build.sh
./build.sh
```

## Photo-SLAM Examples on Some Benchmark Datasets

The benchmark datasets mentioned in our paper: [Replica (NICE-SLAM Version)](https://github.com/cvg/nice-slam), [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download), [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).

0. (optional) Download the dataset.
```
cd scripts
chmod +x ./*.sh
./download_replica.sh
./download_tum.sh
./download_euroc.sh
```

1. For testing, you could use the below commands to run the system after specifying the `PATH_TO_Replica` and `PATH_TO_SAVE_RESULTS`. We would disable the viewer by adding `no_viewer` during the evaluation.
```
../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    PATH_TO_Replica/office0 \
    PATH_TO_SAVE_RESULTS
    # no_viewer 
```

2. We also provide scripts to conduct experiments on all benchmark datasets mentioned in our paper. We ran each sequence five times to lower the effect of the nondeterministic nature of the system. You need to change the dataset root lines in scripts/*.sh then run:
```
cd scripts
chmod +x ./*.sh
./replica_mono.sh
./replica_rgbd.sh
./tum_mono.sh
./tum_rgbd.sh
./euroc_stereo.sh
# etc.
```

3. Evaluation (TODO)
- [ ] evaluation code




## Photo-SLAM Examples with Real Cameras

We provide an example with the Intel RealSense D455 at `examples/realsense_rgbd.cpp`. Please see `scripts/realsense_d455.sh` for running it.





<h3>Citation</h3>
			<pre class="citation-code"><code><span>@inproceedings</span>{hhuang2024photoslam,
	title = {Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras},
	author = {Huang, Huajian and Li, Longwei and Cheng Hui and Yeung, Sai-Kit},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	year = {2024}
}</code></pre>