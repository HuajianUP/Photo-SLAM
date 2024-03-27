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

__forceinline__ __device__ float3 transform_point(
    int idx,
    const float* orig_points,
    const float* transformmatrix)
{
    float3 p_orig = { orig_points[3 * idx],
                      orig_points[3 * idx + 1],
                      orig_points[3 * idx + 2] };

    float3 p_trans = transformPoint4x3(p_orig, transformmatrix);
    return p_trans;
}

__forceinline__ __device__ float3 scale_and_transform_point(
    int idx,
    float scale,
    const float* orig_points,
    const float* transformmatrix)
{
    float3 p_orig = { orig_points[3 * idx],
                      orig_points[3 * idx + 1],
                      orig_points[3 * idx + 2] };

    p_orig.x *= scale;
    p_orig.y *= scale;
    p_orig.z *= scale;

    float3 p_trans = transformPoint4x3(p_orig, transformmatrix);
    return p_trans;
}

__forceinline__ __device__ float4 transfrom_quaternion_using_matrix(
    int idx,
    const float* orig_q,
    const float* transformmatrix)
{
    float4 q_orig = { /*x=*/orig_q[4 * idx + 1],
                      /*y=*/orig_q[4 * idx + 2],
                      /*z=*/orig_q[4 * idx + 3],
                      /*w=*/orig_q[4 * idx] };

    float tx = 2.0f * q_orig.x;
    float ty = 2.0f * q_orig.y;
    float tz = 2.0f * q_orig.z;
    float twx = tx * q_orig.w;
    float twy = ty * q_orig.w;
    float twz = tz * q_orig.w;
    float txx = tx * q_orig.x;
    float txy = ty * q_orig.x;
    float txz = tz * q_orig.x;
    float tyy = ty * q_orig.y;
    float tyz = tz * q_orig.y;
    float tzz = tz * q_orig.z;

    // original Rotation matrix
    float R00 = 1.0f - (tyy + tzz);
    float R01 = txy - twz;
    float R02 = txz + twy;
    float R10 = txy + twz;
    float R11 = 1.0f - (txx + tzz);
    float R12 = tyz - twx;
    float R20 = txz - twy;
    float R21 = tyz + twx;
    float R22 = 1.0f - (txx + tyy);

    // new Rotation matrix
    float R[3][3];
    R[0][0] = transformmatrix[0] * R00 + transformmatrix[4] * R10 + transformmatrix[8] * R20;
    R[0][1] = transformmatrix[0] * R01 + transformmatrix[4] * R11 + transformmatrix[8] * R21;
    R[0][2] = transformmatrix[0] * R02 + transformmatrix[4] * R12 + transformmatrix[8] * R22;
    R[1][0] = transformmatrix[1] * R00 + transformmatrix[5] * R10 + transformmatrix[9] * R20;
    R[1][1] = transformmatrix[1] * R01 + transformmatrix[5] * R11 + transformmatrix[9] * R21;
    R[1][2] = transformmatrix[1] * R02 + transformmatrix[5] * R12 + transformmatrix[9] * R22;
    R[2][0] = transformmatrix[2] * R00 + transformmatrix[6] * R10 + transformmatrix[10] * R20;
    R[2][1] = transformmatrix[2] * R01 + transformmatrix[6] * R11 + transformmatrix[10] * R21;
    R[2][2] = transformmatrix[2] * R02 + transformmatrix[6] * R12 + transformmatrix[10] * R22;

    // new quaternion
    float4 q_trans;
    // This algorithm comes from  "Quaternion Calculus and Fast Animation",
    // Ken Shoemake, 1987 SIGGRAPH course notes
    float t = R[0][0] + R[1][1] + R[2][2]; // trace(R_trans)
    if (t > 0.0f)
    {
        t = sqrt(t + 1.0f);
        q_trans.w = 0.5f * t;
        t = 0.5f / t;
        q_trans.x = (R[2][1] - R[1][2]) * t;
        q_trans.y = (R[0][2] - R[2][0]) * t;
        q_trans.z = (R[1][0] - R[0][1]) * t;
    }
    else
    {
        int i = 0;
        if (R[1][1] > R[0][0])
            i = 1;
        if (R[2][2] > R[i][i])
            i = 2;
        int j = (i + 1) % 3;
        int k = (j + 1) % 3;

        t = sqrt(R[i][i] - R[j][j] - R[k][k] + 1.0f);
        float xyz[3];
        xyz[i] = 0.5f * t;
        t = 0.5f / t;
        q_trans.w = (R[k][j] - R[j][k]) * t;
        xyz[j] = (R[j][i] + R[i][j]) * t;
        xyz[k] = (R[k][i] + R[i][k]) * t;

        q_trans.x = xyz[0];
        q_trans.y = xyz[1];
        q_trans.z = xyz[2];
    }

    return q_trans;
}

__forceinline__ __device__ void insert_point_to_pcd(
    int idx,
    const float3& point,
    float* pcd)
{
    int ptidx = 3 * idx;
    pcd[ptidx] = point.x;
    pcd[ptidx + 1] = point.y;
    pcd[ptidx + 2] = point.z;
}

__forceinline__ __device__ void insert_rot_to_rots(
    int idx,
    const float4& rotq,
    float* rots)
{
    int rotidx = 4 * idx;
    rots[rotidx] = rotq.w;
    rots[rotidx + 1] = rotq.x;
    rots[rotidx + 2] = rotq.y;
    rots[rotidx + 2] = rotq.z;
}
