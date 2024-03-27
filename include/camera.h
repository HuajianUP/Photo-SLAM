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

#pragma once

#include <torch/torch.h>

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/cudawarping.hpp>

#include "types.h"
#include "tensor_utils.h"

class Camera
{
public:
    Camera() {}

    Camera (
        camera_id_t camera_id,
        std::size_t width,
        std::size_t height,
        std::vector<double> params,
        int model_id = 0,
        bool prior_focal_length = true)
        : camera_id_(camera_id),
          width_(width),
          height_(height),
          params_(params),
          model_id_(model_id),
          prior_focal_length_(prior_focal_length)
    {}

    enum CameraModelType{
        INVALID = 0,
        PINHOLE = 1,
        FISHEYE = 2};

public:
    inline void setModelId(const CameraModelType model_id)
    {
        model_id_ = model_id;
        switch (model_id_)
        {
        case PINHOLE: // Pinhole
            params_.resize(4);
            break;

        default:
            break;
        }
    }

    inline void initUndistortRectifyMapAndMask(
        cv::InputArray old_camera_matrix,
        const cv::Size old_size,
        cv::InputArray new_camera_matrix,
        bool do_gaus_pyramid_training)
    {
        cv::initUndistortRectifyMap(
            old_camera_matrix,
            dist_coeff_,
            cv::Mat::eye(3, 3, CV_32F),
            new_camera_matrix,
            cv::Size(this->width_, this->height_),
            CV_32F,
            undistort_map1,
            undistort_map2
        );

        cv::Mat white(old_size, CV_32FC3, cv::Vec3f(1.0f, 1.0f, 1.0f));
        undistortImage(white, undistort_mask);

        if (do_gaus_pyramid_training) {
            assert(!gaus_pyramid_height_.empty() && !gaus_pyramid_width_.empty());
            cv::cuda::GpuMat undistort_mask_gpu;
            undistort_mask_gpu.upload(undistort_mask);
            gaus_pyramid_undistort_mask_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::cuda::GpuMat undistort_mask_gpu_resized;
                cv::cuda::resize(undistort_mask_gpu, undistort_mask_gpu_resized,
                                 cv::Size(gaus_pyramid_width_[l], gaus_pyramid_height_[l]));
                gaus_pyramid_undistort_mask_[l] =
                    tensor_utils::cvGpuMat2TorchTensor_Float32(undistort_mask_gpu_resized);
            }
        }
    }

    inline void undistortImage(cv::InputArray src, cv::OutputArray dst)
    {
        cv::remap(
            src,
            dst,
            undistort_map1,
            undistort_map2,
            cv::InterpolationFlags::INTER_LINEAR
        );
    }

public:
    camera_id_t camera_id_ = 0U;

    int model_id_ = 0;

    std::size_t width_ = 0UL;
    std::size_t height_ = 0UL;

    int num_gaus_pyramid_sub_levels_ = 0;
    std::vector<std::size_t> gaus_pyramid_width_;
    std::vector<std::size_t> gaus_pyramid_height_;

    std::vector<double> params_;

    bool prior_focal_length_ = false;

    float stereo_bf_ = 0.0f;

    cv::Mat dist_coeff_ = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, 0.0f, 0.0f);
    cv::Mat undistort_map1, undistort_map2;
    cv::Mat undistort_mask;
    std::vector<torch::Tensor> gaus_pyramid_undistort_mask_;
};
