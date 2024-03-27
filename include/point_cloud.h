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

#include <Eigen/Core>

class BasicPointCloud
{
public:
    BasicPointCloud() {}
    BasicPointCloud(std::size_t num_points) {
        this->points_.resize(num_points);
        this->colors_.resize(num_points);
        this->normals_.resize(num_points);
    }

public:
    std::vector<Eigen::Vector3f> points_;
    std::vector<Eigen::Vector3f> colors_;
    std::vector<Eigen::Vector3f> normals_;
};
