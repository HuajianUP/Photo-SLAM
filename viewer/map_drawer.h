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

#include "ORB-SLAM3/include/MapPoint.h"
#include "ORB-SLAM3/include/KeyFrame.h"
#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/Settings.h"
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "drawer_utils.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <mutex>

namespace ORB_SLAM3
{

class Settings;

class ImGuiMapDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImGuiMapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings);

    void newParameterLoader(Settings* settings);

    Atlas* mpAtlas;

    void DrawMapPoints();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba);
    void DrawCurrentCamera(glm::mat4 &Twc);
    void SetCurrentCameraTwc(const Sophus::SE3f &Twc);
    void SetCurrentCameraPose(const Sophus::SE3f &Tcw);
    void GetOpenGLCameraMatrix(const bool get_current, Sophus::SE3f &Tcw, glm::mat4 &glmTwc, glm::mat4 &MOw);

    void SetInitCameraTwc(const Sophus::SE3f &Twc);

    bool mbSetInitCamera = false;

private:

    bool ParseViewerParamFile(cv::FileStorage &fSettings);

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    Sophus::SE3f mInitCameraPose;
    Sophus::SE3f mCameraPose;

    std::mutex mMutexCamera;

    float mfFrameColors[6][3] = {{0.0f, 0.0f, 1.0f},
                                {0.8f, 0.4f, 1.0f},
                                {1.0f, 0.2f, 0.4f},
                                {0.6f, 0.0f, 1.0f},
                                {1.0f, 1.0f, 0.0f},
                                {0.0f, 1.0f, 1.0f}};

};

} //namespace ORB_SLAM3
