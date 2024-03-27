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

#include "map_drawer.h"

#include "include/graphics_utils.h"
#include "include/gaussian_mapper.h"

#include "ORB-SLAM3/include/FrameDrawer.h"
#include "ORB-SLAM3/include/MapDrawer.h"
#include "ORB-SLAM3/include/Tracking.h"
#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/include/Settings.h"
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

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

class Tracking;
class FrameDrawer;
class MapDrawer;
class System;
class Settings;

} // namespace ORB_SLAM3

class ImGuiViewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImGuiViewer(
        std::shared_ptr<ORB_SLAM3::System> pSLAM,
        std::shared_ptr<GaussianMapper> pGausMapper,
        bool training = true);
    void readConfigFromFile(std::filesystem::path cfg_path);

    void run();

    bool isStopped();
    void signalStop(const bool going_to_stop = true);

protected:
    void handleUserInput();
    void mouseWheel();
    void mouseDrag();
    void keyboardEvent();

protected:
    std::shared_ptr<ORB_SLAM3::System> pSLAM_;
    std::shared_ptr<GaussianMapper> pGausMapper_;

    ORB_SLAM3::FrameDrawer* pSlamFrameDrawer_;
    ORB_SLAM3::MapDrawer* pSlamMapDrawer_;
    std::shared_ptr<ORB_SLAM3::ImGuiMapDrawer> pMapDrawer_;

    // Status
    bool free_view_enabled_ = true;
    bool init_Twc_set_ = false;
    Sophus::SE3f Tcw_main_, Twc_main_;
    glm::mat4 glmTwc_main_;

    // Configurations
    bool training_ = true;

    int glfw_window_width_, glfw_window_height_;
    int panel_width_, display_panel_height_, training_panel_height_, camera_panel_height_;

    int image_width_, image_height_;
    float SLAM_image_viewer_scale_;
    float viewpointX_ = 0.0f, viewpointY_ = 0.0f, viewpointZ_ = -1.0f, viewpointF_;

    int padded_sub_image_width_;
    int rendered_image_width_, rendered_image_height_;
    float rendered_image_viewer_scale_ = 1.0f;
    int padded_main_image_width_;
    int rendered_image_width_main_, rendered_image_height_main_;
    float rendered_image_viewer_scale_main_ = 1.0f;
    float camera_watch_dist_;

    glm::vec3 up_;
    glm::vec4 up_aligned_;
    glm::vec4 behind_;
    glm::vec3 cam_target_, cam_pos_;
    glm::mat4 cam_proj_;
    glm::mat4 cam_view_;
    glm::mat4 cam_trans_;

    float main_fx_, main_fy_, main_cx_, main_cy_;
    float mouse_left_sensitivity_ = 0.05 * M_PI;
    float mouse_right_sensitivity_ = 0.2 * M_PI;
    float mouse_middle_sensitivity_ = 0.2;
    float keyboard_velocity_ = 0.2;
    float keyboard_anglular_velocity_ = 0.05;

    bool reset_main_to_init_ = false;
    bool tracking_vision_ = false;
    bool show_keyframes_ = false;
    bool show_sparse_mappoints_ = false;
    bool show_main_rendered_ = true;

    float position_lr_init_;
    float feature_lr_;
    float opacity_lr_;
    float scaling_lr_;
    float rotation_lr_;
    float percent_dense_;
    float lambda_dssim_;
    int opacity_reset_interval_;
    float densify_grad_th_;
    int densify_interval_;
    int new_kf_times_of_use_;
    int stable_num_iter_existence_; ///< loop closure correction

    bool keep_training_ = false;
    bool do_gaus_pyramid_training_;
    bool do_inactive_geo_densify_;

    // Status
    bool stopped_ = false;

    // Mutex
    std::mutex mutex_status_;
};
