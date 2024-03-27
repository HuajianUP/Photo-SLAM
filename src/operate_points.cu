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

#include "include/operate_points.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "cuda_rasterizer/operate_points.h"

__global__ void transform_points(
    int P,
    const float* orig_points,
    const float* transformmatrix,
    float* trans_points)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    float3 p_trans = transform_point(idx, orig_points, transformmatrix);
    insert_point_to_pcd(idx, p_trans, trans_points);
}

__global__ void scale_and_transform_points(
    int P,
    const float scale,
    const float* orig_points,
    const float* orig_rots,
    const float* transformmatrix,
    const bool* mask,
    float* trans_points,
    float* trans_rots)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !mask[idx])
        return;

    float3 p_trans = scale_and_transform_point(idx, scale, orig_points, transformmatrix);
    insert_point_to_pcd(idx, p_trans, trans_points);

    float4 rot_trans = transfrom_quaternion_using_matrix(idx, orig_rots, transformmatrix);
    insert_rot_to_rots(idx, rot_trans, trans_rots);
}

void transformPoints(
    torch::Tensor& points,
    torch::Tensor& transformmatrix)
{
    if (points.ndimension() != 2 || points.size(1) != 3) {
        AT_ERROR("points must have dimensions (num_points, 3)");
    }

    const int P = points.size(0);
    torch::Tensor transformed_points = torch::zeros_like(points);

    if (P != 0) {
        transform_points<<<(P + 255) / 256, 256>>>(
            P,
            points.contiguous().data_ptr<float>(),
            transformmatrix.contiguous().data_ptr<float>(),
            transformed_points.contiguous().data_ptr<float>());

        points = transformed_points;
    }
}

void scaleAndTransformThenMarkVisiblePoints(
    torch::Tensor& points,
    torch::Tensor& rots,
    torch::Tensor& point_not_transformed_mask,
    torch::Tensor& point_unstable_mask,
    torch::Tensor& transformmatrix,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix,
    int& num_transformed,
    const float scale)
{
    if (points.ndimension() != 2 || points.size(1) != 3) {
        AT_ERROR("points must have dimensions (num_points, 3)");
    }

    torch::Tensor present = markVisible(
        points,
        viewmatrix,
        projmatrix);

    auto num_points = present.size(0);
    if (point_not_transformed_mask.size(0) !=  num_points || point_unstable_mask.size(0) != num_points) {
        AT_ERROR("points_mask must have dimensions (num_points)");
    }
    torch::Tensor final_mask = torch::logical_and(point_not_transformed_mask, point_unstable_mask);
    final_mask = torch::logical_and(final_mask, present);
    num_transformed += final_mask.sum().item<int>();
    const int P = points.size(0);

    if (P != 0) {
        torch::Tensor transformed_points = torch::zeros_like(points);
        torch::Tensor transformed_rots = torch::zeros_like(rots);

        scale_and_transform_points<<<(P + 255) / 256, 256>>>(
            P,
            scale,
            points.contiguous().data_ptr<float>(),
            rots.contiguous().data_ptr<float>(),
            transformmatrix.contiguous().data_ptr<float>(),
            final_mask.contiguous().data_ptr<bool>(),
            transformed_points.contiguous().data_ptr<float>(),
            transformed_rots.contiguous().data_ptr<float>());

        points.index_put_({final_mask}, transformed_points.index({final_mask}));
        rots.index_put_({final_mask}, transformed_rots.index({final_mask}));
        point_not_transformed_mask.index_put_(
            {final_mask}, torch::full({P}, false, point_not_transformed_mask.options()).index({final_mask}));
    }
}
