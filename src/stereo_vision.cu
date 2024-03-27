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

#include "include/stereo_vision.h"
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
#include "cuda_rasterizer/stereo_vision.h"

__global__ void reproject_depths_pinhole(
    int P,
    const int width,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float* depths,
    const bool* mask,
    float* points)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !mask[idx])
        return;

    int v = idx / width;
    int u = idx - v * width;
    float depth = depths[idx];

    float3 point = reproject_depth_pinhole(
        u, v, depth, fx, fy, cx, cy);
    insert_point_to_pcd(idx, point, points);
}

__global__ void search_neighborhood_to_estimate_depth_and_reproject_pinhole(
    int N,
    int width,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float max_pixel_dist,
    const float* pixels,
    const bool* has3D,
    const float* point3D_orig,
    const float* colors,
    float* point3D_result,
    float* colors_result)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= N)
        return;

    int pxidx = idx * 2;
    float u = pixels[pxidx];
    float v = pixels[pxidx + 1];
    int ptidx = idx * 3;
    int pxidx_in_image = v * width + u;

    if (has3D[idx]) {
        point3D_result[ptidx] = point3D_orig[ptidx];
        point3D_result[ptidx + 1] = point3D_orig[ptidx + 1];
        point3D_result[ptidx + 2] = point3D_orig[ptidx + 2];

        colors_result[ptidx] = colors[pxidx_in_image];
        colors_result[ptidx + 1] = colors[pxidx_in_image + 1];
        colors_result[ptidx + 2] = colors[pxidx_in_image + 2];
        return;
    }

    float min_dist = MAXFLOAT;
    float depth = -1.0f;

    for (int i = 0; i < N; ++i) {
        if (!has3D[i] || i == idx)
            continue;

        // Compute pixel distance between idx and i
        int ipxidx = i * 2;
        float uu = pixels[ipxidx];
        float vv = pixels[ipxidx + 1];
        float u_uu = u - uu;
        float v_vv = v - vv;
        float dist = u_uu * u_uu + v_vv * v_vv;

        if (dist > max_pixel_dist || dist >= min_dist)
            continue;
        // else
        min_dist = dist;
        depth = point3D_orig[i * 3 + 2];
    }

    if (depth > 0.0f) {
        float3 pt_result = reproject_depth_pinhole(
            u, v, depth, fx, fy, cx, cy);

        point3D_result[ptidx] = pt_result.x;
        point3D_result[ptidx + 1] = pt_result.y;
        point3D_result[ptidx + 2] = pt_result.z;

        colors_result[ptidx] = colors[pxidx_in_image];
        colors_result[ptidx + 1] = colors[pxidx_in_image + 1];
        colors_result[ptidx + 2] = colors[pxidx_in_image + 2];
    }
    else {
        point3D_result[ptidx + 2] = -1.0f;
    }
}

torch::Tensor reprojectDepthPinhole(
    torch::Tensor& depth,
    torch::Tensor& mask,
    std::vector<float>& intr,
    int width)
{
    if (depth.ndimension() != 1) {
        AT_ERROR("points must have dimensions (num_points)");
    }

    const int P = depth.size(0);
    torch::Tensor points;

    if(P != 0) {
        points = torch::zeros({P, 3}, depth.options());

        float fx = intr[0];
        float fy = intr[1];
        float cx = intr[2];
        float cy = intr[3];

        reproject_depths_pinhole<<<(P + 255) / 256, 256>>>(
            P, width, fx, fy, cx, cy,
            depth.contiguous().data_ptr<float>(),
            mask.contiguous().data_ptr<bool>(),
            points.contiguous().data_ptr<float>());
    }

    return points;
}

/**
 * @return std::tuple<torch::Tensor, torch::Tensor> <1>pt3D, <2>colors of pt3D
 */
std::tuple<torch::Tensor, torch::Tensor>
monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
    torch::Tensor& kps_pixel,
    torch::Tensor& kps_has3D,
    torch::Tensor& kps_point_local,
    torch::Tensor& colors,
    float max_pixel_dist,
    std::vector<float>& intr,
    int width)
{
    if (kps_pixel.ndimension() != 2 || kps_pixel.size(1) != 2)
        AT_ERROR("kps_pixel must have dimensions (num_points, 2)");
    if (kps_has3D.ndimension() != 1)
        AT_ERROR("kps_has3D must have dimensions (num_points)");
    if (kps_point_local.ndimension() != 2 || kps_point_local.size(1) != 3)
        AT_ERROR("kps_point_local must have dimensions (num_points, 3)");

    int N = kps_pixel.size(0);
    torch::Tensor result_pt, result_color;

    if (N != 0) {
        result_pt = torch::zeros_like(kps_point_local);
        result_color = torch::zeros_like(kps_point_local);

        float fx = intr[0];
        float fy = intr[1];
        float cx = intr[2];
        float cy = intr[3];

        search_neighborhood_to_estimate_depth_and_reproject_pinhole<<<(N + 255) / 256, 256>>>(
            N, width, fx, fy, cx, cy, max_pixel_dist,
            kps_pixel.contiguous().data_ptr<float>(),
            kps_has3D.contiguous().data_ptr<bool>(),
            kps_point_local.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(),
            result_pt.contiguous().data_ptr<float>(),
            result_color.contiguous().data_ptr<float>());

        torch::Tensor depth_valid_flags = torch::where(result_pt.index({torch::indexing::Slice(), 2}) > 0.0f, true, false);
        result_pt = result_pt.index({depth_valid_flags});
        result_color = result_color.index({depth_valid_flags});
    }

    return std::make_tuple(result_pt, result_color);
}