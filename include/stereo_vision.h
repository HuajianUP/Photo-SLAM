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

#include <memory>
#include <vector>

torch::Tensor reprojectDepthPinhole(
    torch::Tensor& depth,
    torch::Tensor& mask,
    std::vector<float>& intr,
    int width);

std::tuple<torch::Tensor, torch::Tensor>
monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
    torch::Tensor& kps_pixel,
    torch::Tensor& kps_has3D,
    torch::Tensor& kps_point_local,
    torch::Tensor& colors,
    float max_pixel_dist,
    std::vector<float>& intr,
    int width);
