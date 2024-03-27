/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <unordered_map>
#include <filesystem>
#include <fstream>

#include <torch/torch.h>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "third_party/colmap/utils/endian.h"
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

void readColmapCamerasBinary(
    std::shared_ptr<GaussianMapper> pMapper,
    const std::filesystem::path& path)
{
    std::ifstream file(path.string(), std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Invalid file path!");

    const std::size_t num_cameras = colmap::ReadBinaryLittleEndian<std::uint64_t>(&file);
    for (std::size_t i = 0; i < num_cameras; ++i) {
        class Camera camera;
        camera.camera_id_ = colmap::ReadBinaryLittleEndian<camera_id_t>(&file);
    
        std::cout << "Loading colmap camera " << camera.camera_id_
            << ", " << (i+1) << "/" << num_cameras << "\r";
        std::cout.flush();

        camera.setModelId(static_cast<Camera::CameraModelType>(colmap::ReadBinaryLittleEndian<int>(&file)));
        camera.width_ = colmap::ReadBinaryLittleEndian<std::uint64_t>(&file);
        camera.height_ = colmap::ReadBinaryLittleEndian<std::uint64_t>(&file);
        colmap::ReadBinaryLittleEndian<double>(&file, &camera.params_);

        camera.num_gaus_pyramid_sub_levels_ = pMapper->num_gaus_pyramid_sub_levels_;
        camera.gaus_pyramid_width_.resize(pMapper->num_gaus_pyramid_sub_levels_);
        camera.gaus_pyramid_height_.resize(pMapper->num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < pMapper->num_gaus_pyramid_sub_levels_; ++l) {
            camera.gaus_pyramid_width_[l] = camera.width_ * pMapper->kf_gaus_pyramid_factors_[l];
            camera.gaus_pyramid_height_[l] = camera.height_ * pMapper->kf_gaus_pyramid_factors_[l];
        }

        cv::Mat K = (
            cv::Mat_<float>(3, 3)
                << camera.params_[0], 0.f, camera.params_[2],
                    0.f, camera.params_[1], camera.params_[3],
                    0.f, 0.f, 1.f
        );
        camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, true);

        pMapper->undistort_mask_[camera.camera_id_] =
            tensor_utils::cvMat2TorchTensor_Float32(
                camera.undistort_mask, pMapper->device_type_);

        cv::Mat viewer_main_undistort_mask;
        int viewer_image_height_main_ = camera.height_ * pMapper->rendered_image_viewer_scale_main_;
        int viewer_image_width_main_ = camera.width_ * pMapper->rendered_image_viewer_scale_main_;
        cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                    cv::Size(viewer_image_width_main_, viewer_image_height_main_));
        pMapper->viewer_main_undistort_mask_[camera.camera_id_] =
            tensor_utils::cvMat2TorchTensor_Float32(
                viewer_main_undistort_mask, pMapper->device_type_);

        if (!pMapper->viewer_camera_id_set_) {
            pMapper->viewer_camera_id_ = camera.camera_id_;
            pMapper->viewer_camera_id_set_ = true;
        }

        pMapper->scene_->addCamera(camera);
    }
    std::cout << std::endl;
}

void readColmapImagesBinary(
    std::shared_ptr<GaussianMapper> pMapper,
    const std::filesystem::path& path,
    const std::filesystem::path& images_dir)
{
    std::ifstream file(path.string(), std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Invalid file path!");

    const std::size_t num_reg_images = colmap::ReadBinaryLittleEndian<std::uint64_t>(&file);
    for (std::size_t i = 0; i < num_reg_images; ++i) {
        std::uint32_t image_id = colmap::ReadBinaryLittleEndian<std::uint32_t>(&file);

        std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(image_id);
        new_kf->zfar_ = pMapper->z_far_;
        new_kf->znear_ = pMapper->z_near_;

        // Pose
        double qw = colmap::ReadBinaryLittleEndian<double>(&file);
        double qx = colmap::ReadBinaryLittleEndian<double>(&file);
        double qy = colmap::ReadBinaryLittleEndian<double>(&file);
        double qz = colmap::ReadBinaryLittleEndian<double>(&file);
        double tx = colmap::ReadBinaryLittleEndian<double>(&file);
        double ty = colmap::ReadBinaryLittleEndian<double>(&file);
        double tz = colmap::ReadBinaryLittleEndian<double>(&file);
        new_kf->setPose(qw, qx, qy, qz, tx, ty, tz);

        // Camera
        camera_id_t camera_id = colmap::ReadBinaryLittleEndian<camera_id_t>(&file);
        Camera& camera = pMapper->scene_->cameras_.at(camera_id);
        new_kf->setCameraParams(camera);

        // Image
        std::string image_name;
        char name_char;
        do {
        file.read(&name_char, 1);
        if (name_char != '\0') {
            image_name += name_char;
        }
        } while (name_char != '\0');
        auto image_path = images_dir / image_name;

        std::cout << "Loading colmap image " << image_id << ", " << image_name
            << ", " << (i+1) << "/" << num_reg_images << "\r";
        std::cout.flush();

        cv::Mat image = cv::imread(image_path.string(), cv::ImreadModes::IMREAD_COLOR);
        cv::cvtColor(image, image, CV_BGR2RGB);
        image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
        cv::Mat imgRGB_undistorted;
        camera.undistortImage(image, imgRGB_undistorted);
        new_kf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image, pMapper->device_type_);
        //TODO: to fit to a size BUG (camera size != image size) in the gaussian splatting tandt dataset
        // new_kf->image_height_ = image.rows;
        // new_kf->image_width_ = image.cols;

        new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        new_kf->gaus_pyramid_times_of_use_ = pMapper->kf_gaus_pyramid_times_of_use_;

        // Points2D
        const std::size_t num_points2D = colmap::ReadBinaryLittleEndian<std::uint64_t>(&file);

        std::vector<Eigen::Vector2d> points2D;
        points2D.reserve(num_points2D);
        std::vector<point3D_id_t> point3D_ids;
        point3D_ids.reserve(num_points2D);
        for (std::size_t j = 0; j < num_points2D; ++j) {
            const double x = colmap::ReadBinaryLittleEndian<double>(&file);
            const double y = colmap::ReadBinaryLittleEndian<double>(&file);
            points2D.emplace_back(x, y);
            point3D_ids.push_back(colmap::ReadBinaryLittleEndian<point3D_id_t>(&file));
        }

        new_kf->setPoints2D(points2D);

        for (point2D_idx_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
            if (point3D_ids[point2D_idx] != std::numeric_limits<point3D_id_t>::max()) {
                new_kf->setPoint3DIdxForPoint2D(point2D_idx, point3D_ids[point2D_idx]);
            }
        }

        new_kf->computeTransformTensors();
        pMapper->scene_->addKeyframe(new_kf, &pMapper->kfid_shuffled_);
        new_kf->img_undist_ = imgRGB_undistorted;
    }

    std::cout << std::endl;
}

void readColmapPoints3DBinary(
    std::shared_ptr<GaussianScene> scene,
    const std::filesystem::path& path)
{
    std::ifstream file(path.string(), std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Invalid file path!");

    const std::size_t num_points3D = colmap::ReadBinaryLittleEndian<std::uint64_t>(&file);
    for (std::size_t i = 0; i < num_points3D; ++i) {
        class Point3D point3D;

        const point3D_id_t point3D_id = colmap::ReadBinaryLittleEndian<point3D_id_t>(&file);

        point3D.xyz_(0) = colmap::ReadBinaryLittleEndian<double>(&file);
        point3D.xyz_(1) = colmap::ReadBinaryLittleEndian<double>(&file);
        point3D.xyz_(2) = colmap::ReadBinaryLittleEndian<double>(&file);
        point3D.color256_(0) = colmap::ReadBinaryLittleEndian<uint8_t>(&file);
        point3D.color256_(1) = colmap::ReadBinaryLittleEndian<uint8_t>(&file);
        point3D.color256_(2) = colmap::ReadBinaryLittleEndian<uint8_t>(&file);
        point3D.color_(0) = point3D.color256_(0) / 255.0f;
        point3D.color_(1) = point3D.color256_(1) / 255.0f;
        point3D.color_(2) = point3D.color256_(2) / 255.0f;
        point3D.error_ = colmap::ReadBinaryLittleEndian<double>(&file);

        const std::size_t track_length = colmap::ReadBinaryLittleEndian<std::uint64_t>(&file);
        for (std::size_t j = 0; j < track_length; ++j) {
            const std::uint32_t image_id = colmap::ReadBinaryLittleEndian<std::uint32_t>(&file);
            const point2D_idx_t point2D_idx = colmap::ReadBinaryLittleEndian<point2D_idx_t>(&file);
        }

        std::cout << "Loading colmap point3D " << point3D_id
            << ", " << (i+1) << "/" << num_points3D << "\r";
        std::cout.flush();

        scene->cachePoint3D(point3D_id, point3D);
    }
    std::cout << std::endl;
}

void readColmapScene(std::shared_ptr<GaussianMapper> pMapper)
{
    auto& model_params = pMapper->getGaussianModelParams();
    auto scene = pMapper->scene_;

    std::filesystem::path cameras_intrinsic_file = model_params.source_path_ / "sparse/0" / "cameras.bin";
    std::filesystem::path cameras_extrinsic_file = model_params.source_path_ / "sparse/0" / "images.bin";
    std::filesystem::path points3D_bin_file      = model_params.source_path_ / "sparse/0" / "points3D.bin";
    std::filesystem::path images_dir             = model_params.source_path_ / "images";

    readColmapCamerasBinary(pMapper, cameras_intrinsic_file);
    readColmapImagesBinary(pMapper, cameras_extrinsic_file, images_dir);
    readColmapPoints3DBinary(scene, points3D_bin_file);
}

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_gaussian_mapping_settings"    /*1*/
                  << " path_to_colmap_data_directory/"       /*2*/
                  << " path_to_output_directory/"            /*3*/
                  << " (optional)no_viewer"                  /*4*/
                  << std::endl;
        return 1;
    }
    bool use_viewer = true;
    if (argc == 5)
        use_viewer = (std::string(argv[4]) == "no_viewer" ? false : true);

    std::string output_directory = std::string(argv[3]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(argv[1]);
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(
            nullptr, gaussian_cfg_path, output_dir, 0, device_type);

    // Read the colmap scene
    pGausMapper->setSensorType(MONOCULAR);
    pGausMapper->setColmapDataPath(argv[2]);
    readColmapScene(pGausMapper);

    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(nullptr, pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    // Train and save results
    pGausMapper->trainColmap();

    if (use_viewer)
        viewer_thd.join();
    return 0;
}