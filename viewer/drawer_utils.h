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

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

/**
 * Note: glm::mat4[col][row]
 */
inline glm::mat4 trans4x4Eigen2glm(const Eigen::Matrix4f &T)
{
    glm::mat4 M;
    for (int i = 0; i < 4; i++)
    {
        M[i][0] = T(0, i);
        M[i][1] = T(1, i);
        M[i][2] = T(2, i);
        M[i][3] = T(3, i);
    }
    // M[0][0] = T(0, 0);
    // M[0][1] = T(1, 0);
    // M[0][2] = T(2, 0);
    // M[0][3] = T(3, 0);

    // M[1][0] = -T(0, 1);
    // M[1][1] = -T(1, 1);
    // M[1][2] = -T(2, 1);
    // M[1][3] = -T(3, 1);

    // M[2][0] = -T(0, 2);
    // M[2][1] = -T(1, 2);
    // M[2][2] = -T(2, 2);
    // M[2][3] = -T(3, 2);

    // M[3][0] = T(0, 3);
    // M[3][1] = -T(1, 3);
    // M[3][2] = -T(2, 3);
    // M[3][3] = T(3, 3);

    return M;
}

inline Sophus::SE3f trans4x4glm2Sophus(const glm::mat4 & M)
{
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    for (int i = 0; i < 3; i++)
    {
        R(0, i) = M[i][0];
        R(1, i) = M[i][1];
        R(2, i) = M[i][2];
        t(i) = M[3][i];
    }

    return Sophus::SE3f(R, t);
}

/**
 * Note: This is a auxiliary inline function, which concerns only the vertices.
 *       It does NOT include glLineWidth, glColor3f, glBegin, glEnd
 */
// inline void glDrawFrameFrustrum(float w, float h, float z, const glm::mat4 &Twc)
// {
    // std::vector<glm::vec4> cam_frustrum(16);
    // cam_frustrum[0] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    // cam_frustrum[1] = glm::vec4(w, h, z, 1.0f);
    // cam_frustrum[2] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    // cam_frustrum[3] = glm::vec4(w, -h, z, 1.0f);
    // cam_frustrum[4] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    // cam_frustrum[5] = glm::vec4(-w, -h, z, 1.0f);
    // cam_frustrum[6] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    // cam_frustrum[7] = glm::vec4(-w, h, z, 1.0f);
    // cam_frustrum[8] = glm::vec4(w, h, z, 1.0f);
    // cam_frustrum[9] = glm::vec4(w, -h, z, 1.0f);
    // cam_frustrum[10] = glm::vec4(-w, h, z, 1.0f);
    // cam_frustrum[11] = glm::vec4(-w, -h, z, 1.0f);
    // cam_frustrum[12] = glm::vec4(-w, h, z, 1.0f);
    // cam_frustrum[13] = glm::vec4(w, h, z, 1.0f);
    // cam_frustrum[14] = glm::vec4(-w, -h, z, 1.0f);
    // cam_frustrum[15] = glm::vec4(w, -h, z, 1.0f);

    // for (int i = 0; i < 16; ++i)
    //     cam_frustrum[i] = Twc * cam_frustrum[i];

    // for (int i = 0; i < 16; ++i)
    //     glVertex3f(cam_frustrum[i].x, cam_frustrum[i].y, cam_frustrum[i].z);
// }