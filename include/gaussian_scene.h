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

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <tuple>
#include <filesystem>

#include "types.h"
#include "camera.h"
#include "point3d.h"
#include "point2d.h"
#include "gaussian_parameters.h"
#include "gaussian_model.h"
#include "gaussian_keyframe.h"

class GaussianScene
{
public:
    GaussianScene(
        GaussianModelParams& args,
        int load_iteration = 0,
        bool shuffle = true,
        std::vector<float> resolution_scales = {1.0f});

public:
    void addCamera(Camera& camera);
    Camera& getCamera(camera_id_t cameraId);

    void addKeyframe(std::shared_ptr<GaussianKeyframe> new_kf, bool* shuffled);
    std::shared_ptr<GaussianKeyframe> getKeyframe(std::size_t fid);
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes();
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> getAllKeyframes();

    void cachePoint3D(point3D_id_t point3D_id, Point3D& point3d);
    Point3D& getPoint3D(point3D_id_t point3DId);
    void clearCachedPoint3D();

    void applyScaledTransformation(
        const float s = 1.0,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));

    std::tuple<Eigen::Vector3f, float> getNerfppNorm();

    std::tuple<std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>,
               std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>>
        splitTrainAndTestKeyframes(const float test_ratio);

public:
    float cameras_extent_; ///< scene_info.nerf_normalization["radius"]

    int loaded_iter_;

    std::map<camera_id_t, Camera> cameras_;
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> keyframes_;
    std::map<point3D_id_t, Point3D> cached_point_cloud_;

protected:
    std::mutex mutex_kfs_;
};
