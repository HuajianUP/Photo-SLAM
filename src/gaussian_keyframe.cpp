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

#include "include/gaussian_keyframe.h"

void GaussianKeyframe::setPose(
    const double qw,
    const double qx,
    const double qy,
    const double qz,
    const double tx,
    const double ty,
    const double tz)
{
    this->R_quaternion_.w() = qw;
    this->R_quaternion_.x() = qx;
    this->R_quaternion_.y() = qy;
    this->R_quaternion_.z() = qz;
    this->R_quaternion_.normalize();
    this->t_.x() = tx;
    this->t_.y() = ty;
    this->t_.z() = tz;

    this->Tcw_ = Sophus::SE3d(this->R_quaternion_, this->t_);

    this->set_pose_ = true;
}

void GaussianKeyframe::setPose(
    const Eigen::Quaterniond& q,
    const Eigen::Vector3d& t)
{
    this->R_quaternion_ = q;
    this->R_quaternion_.normalize();
    this->t_ = t;

    this->Tcw_ = Sophus::SE3d(this->R_quaternion_, this->t_);

    this->set_pose_ = true;
}

Sophus::SE3d GaussianKeyframe::getPose()
{
    return this->Tcw_;
}

Sophus::SE3f GaussianKeyframe::getPosef()
{
    return this->Tcw_.cast<float>();
}

void GaussianKeyframe::setCameraParams(const Camera& camera)
{
    this->camera_id_ = camera.camera_id_;
    this->camera_model_id_ = camera.model_id_;
    this->image_height_ = camera.height_;
    this->image_width_ = camera.width_;

    this->num_gaus_pyramid_sub_levels_ = camera.num_gaus_pyramid_sub_levels_;
    this->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
    this->gaus_pyramid_width_ = camera.gaus_pyramid_width_;

    this->intr_.resize(camera.params_.size());
    for (std::size_t i = 0; i < camera.params_.size(); ++i)
        this->intr_[i] = static_cast<float>(camera.params_[i]);

    switch (this->camera_model_id_)
    {
    case 1: // Pinhole
    {
        float focal_length_x = static_cast<float>(camera.params_[0]);
        float focal_length_y = static_cast<float>(camera.params_[1]);
        this->FoVx_ = graphics_utils::focal2fov(focal_length_x, camera.width_);
        this->FoVy_ = graphics_utils::focal2fov(focal_length_y, camera.height_);
        this->set_camera_ = true;
    }
    break;

    default:
    {
        throw std::runtime_error("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!");
    }
    break;
    }
}

void GaussianKeyframe::setPoints2D(const std::vector<Eigen::Vector2d>& points2D)
{
    this->points2D_.clear();
    auto num_points2D = points2D.size();
    this->points2D_.resize(num_points2D);
    for (point2D_idx_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
        points2D_[point2D_idx].xy_ = points2D[point2D_idx];
    }
}

void GaussianKeyframe::setPoint3DIdxForPoint2D(
    const point2D_idx_t point2D_idx,
    const point3D_id_t point3D_id)
{
    points2D_.at(point2D_idx).point3D_id_ = point3D_id;
}

void GaussianKeyframe::computeTransformTensors()
{
    if (this->set_pose_ && this->set_camera_) {
        this->world_view_transform_ = tensor_utils::EigenMatrix2TorchTensor(
            this->getWorld2View2(this->trans_, this->scale_),
            torch::kCUDA
        ).transpose(0, 1);

        if (!this->set_projection_matrix_) {
            this->projection_matrix_ = this->getProjectionMatrix(
                this->znear_,
                this->zfar_,
                this->FoVx_,
                this->FoVy_,
                torch::kCUDA
            ).transpose(0, 1);
            this->set_projection_matrix_ = true;
        }

        this->full_proj_transform_ = (this->world_view_transform_.unsqueeze(0).bmm(
            this->projection_matrix_.unsqueeze(0))).squeeze(0);

        this->camera_center_ = this->world_view_transform_.inverse().index({3, torch::indexing::Slice(0, 3)});
    }
    else if (!this->set_pose_ && this->set_camera_) {
        std::cerr << "Could not compute transform tensors for keyframe " << this->fid_ << " because POSE is not set!" << std::endl;
    }
    else if (!this->set_camera_) {
        std::cerr << "Could not compute transform tensors for keyframe " << this->fid_ << " because CAMERA is not set!" << std::endl;
    }
    else {
        std::cerr << "Could not compute transform tensors for keyframe " << this->fid_ << " because POSE and CAMERA are not set!" << std::endl;
    }
}

Eigen::Matrix4f
GaussianKeyframe::getWorld2View2(
    const Eigen::Vector3f& trans,
    float scale)
{
    Eigen::Matrix4f Rt;
    Rt.setZero();
    Eigen::Matrix3f R = this->R_quaternion_.toRotationMatrix().cast<float>();
    Rt.topLeftCorner<3, 3>() = R;
    Eigen::Vector3f t = this->t_.cast<float>();
    Rt.topRightCorner<3, 1>() = t;
    Rt(3, 3) = 1.0f;

    Eigen::Matrix4f C2W = Rt.inverse();
    Eigen::Vector3f cam_center = C2W.block<3, 1>(0, 3);
    cam_center += trans;
    cam_center *= scale;
    C2W.block<3, 1>(0, 3) = cam_center;
    Rt = C2W.inverse();
    return Rt;
}

torch::Tensor
GaussianKeyframe::getProjectionMatrix(
    float znear,
    float zfar,
    float fovX,
    float fovY,
    torch::DeviceType device_type)
{
    float tanHalfFovY = std::tan(fovY / 2);
    float tanHalfFovX = std::tan(fovX / 2);

    float top = tanHalfFovY * znear;
    float bottom = -top;
    float right = tanHalfFovX * znear;
    float left = -right;

    torch::Tensor P = torch::zeros({4, 4}, torch::TensorOptions().device(device_type));

    float z_sign = 1.0f;

    P.index({0, 0}) = 2.0 * znear / (right - left);
    P.index({1, 1}) = 2.0 * znear / (top - bottom);
    P.index({0, 2}) = (right + left) / (right - left);
    P.index({1, 2}) = (top + bottom) / (top - bottom);
    P.index({3, 2}) = z_sign;
    P.index({2, 2}) = z_sign * zfar / (zfar - znear);
    P.index({2, 3}) = -(zfar * znear) / (zfar - znear);
    return P;
}

int GaussianKeyframe::getCurrentGausPyramidLevel()
{
    for (int i = 0; i < gaus_pyramid_times_of_use_.size(); ++i) {
        if (gaus_pyramid_times_of_use_[i]) {
            --gaus_pyramid_times_of_use_[i];
            return i;
        }
    }
    // If all sub levels has been used up
    return num_gaus_pyramid_sub_levels_;
}
