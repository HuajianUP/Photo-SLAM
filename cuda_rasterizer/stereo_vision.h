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

#include "config.h"
#include "stdio.h"
#include "auxiliary.h"

#ifndef BLOCK_X
    #define BLOCK_X 16
#endif

#ifndef BLOCK_Y
    #define BLOCK_Y 16
#endif

#ifndef BLOCK_SIZE
    #define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#endif

#ifndef NUM_WARPS
    #define NUM_WARPS (BLOCK_SIZE/32)
#endif

__forceinline__ __device__ float3 reproject_depth_pinhole(
    const int u,
    const int v,
    const float depth,
    const float fx,
    const float fy,
    const float cx,
    const float cy)
{
    float3 pt;
    pt.x = (u - cx) * depth / fx;
    pt.y = (v - cy) * depth / fy;
    pt.z = depth;
    return pt;
}
