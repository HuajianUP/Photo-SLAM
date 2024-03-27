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

#include "imgui_viewer.h"

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "[ImGuiViewer]GLFW Error %d: %s\n", error, description);
}

ImGuiViewer::ImGuiViewer(
    std::shared_ptr<ORB_SLAM3::System> pSLAM,
    std::shared_ptr<GaussianMapper> pGausMapper,
    bool training)
    : glfw_window_width_(1600),
      glfw_window_height_(900),
      panel_width_(372),
      display_panel_height_(144),
      training_panel_height_(440),
      camera_panel_height_(144),
      SLAM_image_viewer_scale_(1.0f),
      training_(training)
{
    this->pSLAM_ = pSLAM;
    this->pGausMapper_ = pGausMapper;

    cv::Size im_size;
    if (pSLAM)
    {
        // ORB_SLAM3 settings
        ORB_SLAM3::Settings* settings = pSLAM->getSettings();

        im_size = settings->newImSize();
        image_height_ = im_size.height;
        image_width_ = im_size.width;

        viewpointX_ = settings->viewPointX();
        viewpointY_ = settings->viewPointY();
        viewpointZ_ = settings->viewPointZ();
        viewpointF_ = settings->camera1()->getParameter(1);
    }
    else
    {
        image_height_ = pGausMapper->scene_->cameras_.begin()->second.height_;
        image_width_ = pGausMapper->scene_->cameras_.begin()->second.width_;
        viewpointF_ = pGausMapper->scene_->cameras_.begin()->second.params_[1];
    }

    main_fx_ = pGausMapper->scene_->cameras_.begin()->second.params_[0];
    main_fy_ = pGausMapper->scene_->cameras_.begin()->second.params_[1];

    // Gaussian Mapper settings
    std::filesystem::path cfg_file_path = pGausMapper->config_file_path_;
    readConfigFromFile(cfg_file_path);
    SLAM_image_viewer_scale_ = static_cast<float>(rendered_image_width_) / image_width_;

    float fovy = graphics_utils::focal2fov(viewpointF_, im_size.height);
    cam_proj_ = glm::perspective(
        fovy < M_PIf32 ? fovy : M_PIf32, (float)glfw_window_width_ / (float)glfw_window_height_, 0.01f, 100.0f);

    up_ = glm::vec3(0.0f, -1.0f, 0.0f);
    up_aligned_ = glm::vec4(up_, 1.0f);
    behind_ = glm::vec4(0.0f, 0.0f, -camera_watch_dist_, 1.0f);
    cam_pos_ = glm::vec3(viewpointX_, viewpointY_, viewpointZ_);
    cam_target_ = glm::vec3(0.0f, 0.0f, 0.0f);
    cam_view_ = glm::lookAt(cam_pos_, cam_target_, up_);
    cam_trans_ = cam_proj_ * cam_view_;

    // Create drawers
    if (pSLAM)
    {
        pSlamFrameDrawer_ = pSLAM->getFrameDrawer();
        pSlamMapDrawer_ = pSLAM->getMapDrawer();
        pMapDrawer_ = std::make_shared<ORB_SLAM3::ImGuiMapDrawer>(
            pSLAM->getAtlas(), std::string(), pSLAM->getSettings());
    }
}

void ImGuiViewer::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened())
       throw std::runtime_error("[ImGuiViewer]Failed to open settings file at: " + cfg_path.string());
    std::cout << "[ImGuiViewer]Reading parameters from " << cfg_path << std::endl;

    glfw_window_width_ =
        settings_file["GaussianViewer.glfw_window_width"].operator int();
    glfw_window_height_ =
        settings_file["GaussianViewer.glfw_window_height"].operator int();
    main_cx_ = glfw_window_width_ / 2;
    main_cy_ = glfw_window_height_ / 2;

    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_height_ = image_height_ * rendered_image_viewer_scale_;
    rendered_image_width_ = image_width_ * rendered_image_viewer_scale_;

    int temp = rendered_image_width_ % 4;
    padded_sub_image_width_ = rendered_image_width_ + 4 - (temp == 0 ? 4 : temp);

    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();
    rendered_image_height_main_ = image_height_ * rendered_image_viewer_scale_main_;
    rendered_image_width_main_ = image_width_ * rendered_image_viewer_scale_main_;

    temp = rendered_image_width_main_ % 4;
    padded_main_image_width_ = rendered_image_width_main_ + 4 - (temp == 0 ? 4 : temp); 

    camera_watch_dist_ =
        settings_file["GaussianViewer.camera_watch_dist"].operator float();

    // Initialize configurations same as the GaussianMapper
    position_lr_init_ = pGausMapper_->positionLearningRateInit();
    feature_lr_ = pGausMapper_->featureLearningRate();
    opacity_lr_ = pGausMapper_->opacityLearningRate();
    scaling_lr_ = pGausMapper_->scalingLearningRate();
    rotation_lr_ = pGausMapper_->rotationLearningRate();
    percent_dense_ = pGausMapper_->percentDense();
    lambda_dssim_ = pGausMapper_->lambdaDssim();
    opacity_reset_interval_ = pGausMapper_->opacityResetInterval();
    densify_grad_th_ = pGausMapper_->densifyGradThreshold();
    densify_interval_ = pGausMapper_->densifyInterval();
    new_kf_times_of_use_ = pGausMapper_->newKeyframeTimesOfUse();
    stable_num_iter_existence_ = pGausMapper_->stableNumIterExistence();

    do_gaus_pyramid_training_ = pGausMapper_->isdoingGausPyramidTraining();
    do_inactive_geo_densify_ = pGausMapper_->isdoingInactiveGeoDensify();
}

void ImGuiViewer::run()
{
    // Initialize glfw
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        throw std::runtime_error("[ImGuiViewer]Fails to initialize!");

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // Create window with graphics context
    GLFWwindow* window =
        glfwCreateWindow(glfw_window_width_, glfw_window_height_,
                         "Photo-SLAM", nullptr, nullptr);
    if (window == nullptr)
        throw std::runtime_error("[ImGuiViewer]Fails to create window!");
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    glEnable(GL_DEPTH_TEST); // Enable 3D Mouse handler

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Variables for tracking
    Sophus::SE3f Tcw, TcwInit;
    cam_pos_ = glm::vec3(viewpointX_, viewpointY_, viewpointZ_);
    glm::vec4 cam_pos_aligned = glm::vec4(cam_pos_, 1.0f);
    cam_target_ = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cam_direction = cam_pos_ - cam_target_;
    glm::vec3 cam_right = glm::normalize(glm::cross(up_, cam_direction));
    glm::vec3 cam_up = glm::cross(cam_direction, cam_right);
    glm::vec4 cam_up_aligned = glm::vec4(cam_up, 1.0f);
    cam_view_ = glm::lookAt(cam_pos_, cam_target_, cam_up);
    glm::mat4 glmTwc, Twr, glmTwcInit;
    glmTwc = glm::mat4(1.0f);
    glmTwcInit = glm::mat4(1.0f);
    glmTwc_main_ = glm::mat4(1.0f);
    glm::mat4 Ow, OwInit;
    Ow = glm::mat4(1.0f);
    OwInit = glm::mat4(1.0f);

    // Variables for showing images
    cv::Rect image_rect_sub(0, 0, rendered_image_width_, rendered_image_height_);
    cv::Rect image_rect_main(0, 0, rendered_image_width_main_, rendered_image_height_main_);

    GLuint SLAM_img_texture, rendered_img_texture, main_img_texture;

    glGenTextures(1, &SLAM_img_texture);
    glBindTexture(GL_TEXTURE_2D, SLAM_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &rendered_img_texture);
    glBindTexture(GL_TEXTURE_2D, rendered_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &main_img_texture);
    glBindTexture(GL_TEXTURE_2D, main_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Main loop
    while(!isStopped() && !glfwWindowShouldClose(window))
    {
        //--------------Poll and handle events (inputs, window resize, etc.)--------------
        glfwPollEvents();

        //--------------Start the Dear ImGui frame--------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //--------------Get pose of current tracked frame--------------
        if (pSLAM_)
        {
            if (!pMapDrawer_->mbSetInitCamera)
            {
                Sophus::SE3f initTwc = pSlamMapDrawer_->GetCurrentCameraPose();
                pMapDrawer_->SetInitCameraTwc(initTwc);
                pMapDrawer_->SetCurrentCameraTwc(initTwc);
                pMapDrawer_->mbSetInitCamera = true;
            }
            else
            {
                pMapDrawer_->SetCurrentCameraTwc(pSlamMapDrawer_->GetCurrentCameraPose());
            }
            pMapDrawer_->GetOpenGLCameraMatrix(true, Tcw, glmTwc, Ow);
            if (!init_Twc_set_)
                pMapDrawer_->GetOpenGLCameraMatrix(false, TcwInit, glmTwcInit, OwInit);
        }
        if (tracking_vision_)
        {
            glm::vec3 cam_target = glm::vec3(Ow[3][0], Ow[3][1], Ow[3][2]);
            cam_pos_aligned = glmTwc * behind_;
            glm::vec3 cam_pos = glm::vec3(cam_pos_aligned.x, cam_pos_aligned.y, cam_pos_aligned.z);
            cam_direction = cam_pos - cam_target;
            // cam_right = glm::normalize(glm::cross(up_, cam_direction));
            // cam_up = glm::cross(cam_direction, cam_right);
            cam_up_aligned = glmTwc * up_aligned_;
            cam_up = glm::normalize(glm::vec3(cam_up_aligned.x, cam_up_aligned.y, cam_up_aligned.z) - cam_target);
            cam_right = glm::normalize(glm::cross(cam_up, cam_direction));
            cam_up = glm::cross(cam_direction, cam_right);
            cam_view_ = glm::lookAt(cam_pos, cam_target, cam_up);
            cam_trans_ = cam_proj_ * cam_view_;
        }
        else
        {
            if (reset_main_to_init_ || !init_Twc_set_)
            {
                cam_target_ = glm::vec3(OwInit[3][0], OwInit[3][1], OwInit[3][2]);
                cam_pos_aligned = glmTwcInit * behind_;
                glmTwc_main_ = glmTwcInit;
                Tcw_main_ = TcwInit;
                Twc_main_ = Tcw_main_.inverse();
                init_Twc_set_ = true;
                reset_main_to_init_ = false;
            }
            else
            {
                Tcw_main_ = trans4x4glm2Sophus(glmTwc_main_).inverse();
                handleUserInput();
                glmTwc_main_ = trans4x4Eigen2glm(Tcw_main_.inverse().matrix());
                cam_target_ = glm::vec3(glmTwc_main_[3][0], glmTwc_main_[3][1], glmTwc_main_[3][2]);
                cam_pos_aligned = glmTwc_main_ * behind_;
            }
            cam_pos_ = glm::vec3(cam_pos_aligned.x, cam_pos_aligned.y, cam_pos_aligned.z);
            cam_direction = cam_pos_ - cam_target_;
            // cam_right = glm::normalize(glm::cross(up_, cam_direction));
            // cam_up = glm::cross(cam_direction, cam_right);
            cam_up_aligned = glmTwc_main_ * up_aligned_;
            cam_up = glm::normalize(glm::vec3(cam_up_aligned.x, cam_up_aligned.y, cam_up_aligned.z) - cam_target_);
            cam_right = glm::normalize(glm::cross(cam_up, cam_direction));
            cam_up = glm::cross(cam_direction, cam_right);
            cam_view_ = glm::lookAt(cam_pos_, cam_target_, cam_up);
            cam_trans_ = cam_proj_ * cam_view_;
        }

        if (pSLAM_)
        {
            //--------------Draw SLAM frame image--------------
            cv::Mat SLAM_img_to_show;
            cv::Mat SLAM_img_with_text = pSlamFrameDrawer_->DrawFrame(1.0f);
            if (SLAM_image_viewer_scale_ != 1.0f)
            {
                int width = rendered_image_width_;
                int height = static_cast<int>(SLAM_img_with_text.rows * SLAM_image_viewer_scale_);
                cv::resize(SLAM_img_with_text, SLAM_img_with_text, cv::Size(width, height));
                SLAM_img_to_show = cv::Mat(height, padded_sub_image_width_, CV_8UC3, cv::Vec3f(0, 0, 0));
            }
            else
            {
                SLAM_img_to_show = cv::Mat(image_height_, padded_sub_image_width_, CV_8UC3, cv::Vec3f(0, 0, 0));
            }
            cv::Rect SLAM_image_rect(0, 0, SLAM_img_with_text.cols, SLAM_img_with_text.rows);
            SLAM_img_with_text.copyTo(SLAM_img_to_show(SLAM_image_rect));
            // Upload SLAM frame
            glBindTexture(GL_TEXTURE_2D, SLAM_img_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SLAM_img_to_show.cols, SLAM_img_to_show.rows,
                        0, GL_BGR, GL_UNSIGNED_BYTE, (unsigned char*)SLAM_img_to_show.data);
            // Create an ImGui window to show the SLAM frame
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(rendered_image_width_ + 12, SLAM_img_to_show.rows + 40), ImGuiCond_Once);
            {
                ImGui::Begin("SLAM Frame");
                ImGui::Image((void *)(intptr_t)SLAM_img_texture,
                            ImVec2(SLAM_img_to_show.cols, SLAM_img_to_show.rows));
                ImGui::End();
            }

            //--------------Draw current gaussian mapper frame image--------------
            // Render gaussian mapper frame
            cv::Mat rendered_img = pGausMapper_->renderFromPose(
                Tcw, rendered_image_width_, rendered_image_height_, false);
            cv::Mat rendered_img_to_show = cv::Mat(rendered_image_height_, padded_sub_image_width_, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
            rendered_img.copyTo(rendered_img_to_show(image_rect_sub));
            // Upload rendered frame
            glBindTexture(GL_TEXTURE_2D, rendered_img_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rendered_img_to_show.cols, rendered_img_to_show.rows,
                        0, GL_RGB, GL_FLOAT, (float*)rendered_img_to_show.data);
            // Create an ImGui window to show the rendered frame
            ImGui::SetNextWindowPos(ImVec2(0, SLAM_img_to_show.rows + 40), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(rendered_image_width_ + 12, rendered_img_to_show.rows + 40), ImGuiCond_Once);
            {
                ImGui::Begin("Current Rendered Frame");
                ImGui::Image((void *)(intptr_t)rendered_img_texture,
                            ImVec2(rendered_img_to_show.cols, rendered_img_to_show.rows));
                ImGui::End();
            }
        }

        //--------------Draw main window image--------------
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        // Draw main window image
        if (show_main_rendered_)
        {
            auto drawlist = ImGui::GetBackgroundDrawList();
            if (pSLAM_ && tracking_vision_)
            {
                drawlist->AddImage((void *)(intptr_t)rendered_img_texture, ImVec2(0, 0),
                                   ImVec2(glfw_window_width_, glfw_window_height_));
            }
            else
            {
                cv::Mat main_img = pGausMapper_->renderFromPose(
                    Tcw_main_, rendered_image_width_main_, rendered_image_height_main_, true);
                cv::Mat main_img_to_show = cv::Mat(rendered_image_height_main_, padded_main_image_width_, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
                main_img.copyTo(main_img_to_show(image_rect_main));
                glBindTexture(GL_TEXTURE_2D, main_img_texture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, main_img_to_show.cols, main_img_to_show.rows,
                     0, GL_RGB, GL_FLOAT, (float*)main_img_to_show.data);
                drawlist->AddImage((void *)(intptr_t)main_img_texture, ImVec2(0, 0),
                                   ImVec2(glfw_window_width_, glfw_window_height_));
            }
        }
        //--------------Get current parameters--------------
        VariableParameters params_in = pGausMapper_->getVaribleParameters();
        position_lr_init_ = params_in.position_lr_init;
        feature_lr_ = params_in.feature_lr;
        opacity_lr_ = params_in.opacity_lr;
        scaling_lr_ = params_in.scaling_lr;
        rotation_lr_ = params_in.rotation_lr;
        percent_dense_ = params_in.percent_dense;
        lambda_dssim_ = params_in.lambda_dssim;
        opacity_reset_interval_ = params_in.opacity_reset_interval;
        densify_grad_th_ = params_in.densify_grad_th;
        densify_interval_ = params_in.densify_interval;
        new_kf_times_of_use_ = params_in.new_kf_times_of_use;
        stable_num_iter_existence_ = params_in.stable_num_iter_existence;
        keep_training_ = params_in.keep_training;
        do_gaus_pyramid_training_ = params_in.do_gaus_pyramid_training;
        do_inactive_geo_densify_ = params_in.do_inactive_geo_densify;

        //--------------Display mode panel--------------
        ImGui::SetNextWindowPos(ImVec2(glfw_window_width_ - panel_width_, 0), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(panel_width_, display_panel_height_), ImGuiCond_Once);
        {
            ImGui::Begin("Display Mode");

            if (training_)
            {
                ImGui::Checkbox("Tracking vision", &tracking_vision_);
                ImGui::Checkbox("Show KeyFrames", &show_keyframes_);
                ImGui::Checkbox("Show sparse MapPoints", &show_sparse_mappoints_);
            }
            ImGui::Checkbox("Show main window rendered", &show_main_rendered_);

            ImGui::Text("Viewer average FPS %.1f", io.Framerate);
            ImGui::End();
        }

        //--------------Training options panel--------------
        if (training_)
        {
            ImGui::SetNextWindowPos(ImVec2(glfw_window_width_ - panel_width_, display_panel_height_ + 8), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(panel_width_, training_panel_height_), ImGuiCond_Once);
            {
                ImGui::Begin("Training Options");

                ImGui::Text("Iteration: %d", pGausMapper_->getIteration());

                ImGui::Checkbox("Gaussian-pyramid-based training", &do_gaus_pyramid_training_);
                ImGui::Checkbox("Densify with inactive geometries", &do_inactive_geo_densify_);
                ImGui::Checkbox("Keep training after stop", &keep_training_);

                ImGui::SliderFloat("Position l.r.", &position_lr_init_, 0.00001f, 0.00100f, "%.5f");
                ImGui::SliderFloat("Feature l.r.", &feature_lr_, 0.0001f, 0.0050f, "%.5f");
                ImGui::SliderFloat("Opacity l.r.", &opacity_lr_, 0.01f, 0.10f, "%.5f");
                ImGui::SliderFloat("Scaling l.r.", &scaling_lr_, 0.001f, 0.010f, "%.5f");
                ImGui::SliderFloat("Rotation l.r.", &rotation_lr_, 0.0001f, 0.0100f, "%.5f");
                ImGui::SliderFloat("Percent dense", &percent_dense_, 0.001f, 0.100f, "%.3f");
                ImGui::SliderFloat("Lambda dssim", &lambda_dssim_, 0.01f, 0.40f, "%.2f");
                ImGui::SliderInt("Opacity reset", &opacity_reset_interval_, 0, 6000);
                ImGui::SliderFloat("Densify grad th.", &densify_grad_th_, 0.0001f, 0.0020f, "%.5f");
                ImGui::SliderInt("Densify int.", &densify_interval_, 1, 400);
                ImGui::SliderInt("New kf. using", &new_kf_times_of_use_, 0, 10);
                ImGui::SliderInt("Stable iter.", &stable_num_iter_existence_, 0, 100);

                ImGui::End();
            }
        }
        VariableParameters params_out;
        params_out.position_lr_init = position_lr_init_;
        params_out.feature_lr = feature_lr_;
        params_out.opacity_lr = opacity_lr_;
        params_out.scaling_lr = scaling_lr_;
        params_out.rotation_lr = rotation_lr_;
        params_out.percent_dense = percent_dense_;
        params_out.lambda_dssim = lambda_dssim_;
        params_out.opacity_reset_interval = opacity_reset_interval_;
        params_out.densify_grad_th = densify_grad_th_;
        params_out.densify_interval = densify_interval_;
        params_out.new_kf_times_of_use = new_kf_times_of_use_;
        params_out.stable_num_iter_existence = stable_num_iter_existence_;
        params_out.keep_training = keep_training_;
        params_out.do_gaus_pyramid_training = do_gaus_pyramid_training_;
        params_out.do_inactive_geo_densify = do_inactive_geo_densify_;
        pGausMapper_->setVaribleParameters(params_out);

        //--------------Camera view panel--------------
        ImGui::SetNextWindowPos(ImVec2(glfw_window_width_ - panel_width_, (training_ ? display_panel_height_ + training_panel_height_ + 16 : display_panel_height_ + 8)), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(panel_width_, camera_panel_height_), ImGuiCond_Once);
        {
            ImGui::Begin("Camera View Velocity");

            ImGui::SliderFloat("Mouse Left", &mouse_left_sensitivity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Mouse Right", &mouse_right_sensitivity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Mouse Middle", &mouse_middle_sensitivity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Keyboard t", &keyboard_velocity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Keyboard R", &keyboard_anglular_velocity_, 0.01f, 1.0f, "%.2f");

            ImGui::End();
        }

        //--------------ImGui Rendering--------------
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        //--------------Draw main window SLAM--------------
        // Set relative viewpoint
        glPushMatrix();
        glMultMatrixf(&cam_trans_[0][0]);
        // Draw camera, KeyFrames and MapPoints
        if (pSLAM_ && show_keyframes_)
        {
            pMapDrawer_->DrawCurrentCamera(tracking_vision_ ? glmTwc : glmTwc_main_);
            pMapDrawer_->DrawKeyFrames(true, false, true, false);
        }
        if (pSLAM_ && show_sparse_mappoints_)
        {
            pMapDrawer_->DrawMapPoints();
        }
        // Clear relative viewpoint
        glPopMatrix();

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (!keep_training_  && pGausMapper_->isStopped())
            signalStop();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    if (pSLAM_ && !pSLAM_->isShutDown())
        pSLAM_->Shutdown();
    else
        pGausMapper_->signalStop();

    if (pGausMapper_->isKeepingTraining())
        pGausMapper_->setKeepTraining(false);
}

bool ImGuiViewer::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void ImGuiViewer::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

/**
 * We modify Twc_main_ then Tcw_main_ (Sophus::SE3f) to handle mouse and keyboard inputs
 */
void ImGuiViewer::handleUserInput()
{
    if (tracking_vision_)
    {
        free_view_enabled_ = false;
        return;
    }
    else
    {
        free_view_enabled_ = true;
    }

    Twc_main_ = Tcw_main_.inverse();

    // Only respond to mouse inputs when not interacting with ImGui
    if (!ImGui::IsAnyItemActive() && !ImGui::GetIO().WantCaptureMouse)
    {
        mouseWheel();
        mouseDrag();
    }

    // Respond to keyboard inputs
    keyboardEvent();

    Tcw_main_ = Twc_main_.inverse();
}

void ImGuiViewer::mouseWheel()
{
    float delta = ImGui::GetIO().MouseWheel;

    if (delta == 0)
        return;

    float scale_factor = std::pow(1.1f, -delta);

    // float mouse_x = ImGui::GetMousePos().x;
    // float mouse_y = ImGui::GetMousePos().y;

    // float focus3D_x = (mouse_x - main_cx_) / main_fx_;
    // float focus3D_y = (mouse_y - main_cy_) / main_fy_;

    //---Translation---
    // Eigen::Vector3f translating = Eigen::Vector3f::Zero();
    // translating.x() += focus3D_x * (1 - scale_factor);
    // translating.y() += focus3D_y * (1 - scale_factor);
    //-------------

    //---Apply---
    Eigen::Matrix3f R = Twc_main_.rotationMatrix();
    Twc_main_.translation() *= scale_factor;
    // Twc_main_.translation() += (R * translating);
}

void ImGuiViewer::mouseDrag()
{
    float delta_rel_x = ImGui::GetIO().MouseDelta.x / glfw_window_width_;
    float delta_rel_y = ImGui::GetIO().MouseDelta.y / glfw_window_height_;
    float delta_l = delta_rel_x * delta_rel_x + delta_rel_y * delta_rel_y;

    //---Rotation---
    Eigen::Vector3f eulars = Eigen::Vector3f::Zero();
    // Left held
    if (ImGui::GetIO().MouseDown[0])
    {
        // Upward (delta_y < 0): pitch upward (Rx+)
        eulars.x() -= M_PI * delta_rel_y;
        // Leftward (delta_x < 0): yaw leftward (Ry-)
        eulars.y() += M_PI * delta_rel_x;
        // Angular Velocity
        eulars.x() *= mouse_left_sensitivity_;
        eulars.y() *= mouse_left_sensitivity_;
    }

    // Right held
    if (ImGui::GetIO().MouseDown[1])
    {
        // Leftward (delta_x < 0): roll counterclockwise (Rz-)
        eulars.z() += M_PI * (delta_rel_x < 0.0f ? -delta_l : delta_l);
        // Angular Velocity
        eulars.z() *= mouse_right_sensitivity_;
    }
    // To rotation matrix
    Eigen::AngleAxisf roll_angle(eulars.z(), Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yaw_angle(eulars.y(), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitch_angle(eulars.x(), Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = roll_angle * yaw_angle * pitch_angle;
    Eigen::Matrix3f rotating = q.matrix();
    //-------------

    //---Translation---
    // Middle held
    Eigen::Vector3f translating = Eigen::Vector3f::Zero();
    if (ImGui::GetIO().MouseDown[2])
    {
        translating.x() += delta_rel_x;
        translating.y() -= delta_rel_y;
        // Velocity
        translating *= mouse_middle_sensitivity_;
    }
    //-------------

    //---Apply---
    Eigen::Matrix3f R = Twc_main_.rotationMatrix();
    Twc_main_.translation() += (R * translating);
    Twc_main_.setRotationMatrix(R * rotating);
}

void ImGuiViewer::keyboardEvent()
{
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;

    // R: Reset camera view to init
    if (ImGui::IsKeyPressed(ImGuiKey_R))
        reset_main_to_init_ = true;

    //---Translation---
    Eigen::Vector3f translating = Eigen::Vector3f::Zero();
    // W: forward
    if (ImGui::IsKeyDown(ImGuiKey_W))
        translating.z() += 1.0f;
    // S: backward
    if (ImGui::IsKeyDown(ImGuiKey_S))
        translating.z() -= 1.0f;
    // A: leftward
    if (ImGui::IsKeyDown(ImGuiKey_A))
        translating.x() -= 1.0f;
    // D: rightward
    if (ImGui::IsKeyDown(ImGuiKey_D))
        translating.x() += 1.0f;
    // Velocity
    translating *= keyboard_velocity_;
    //-------------

    //---Rotation---
    Eigen::Vector3f eulars = Eigen::Vector3f::Zero();
    // I: pitch upward (Rx+)
    if (ImGui::IsKeyDown(ImGuiKey_I))
        eulars.x() += M_PI;
    // K: pitch downward (Rx-)
    if (ImGui::IsKeyDown(ImGuiKey_K))
        eulars.x() -= M_PI;
    // J: yaw leftward (Ry-)
    if (ImGui::IsKeyDown(ImGuiKey_J))
        eulars.y() -= M_PI;
    // L: yaw rightward (Ry+)
    if (ImGui::IsKeyDown(ImGuiKey_L))
        eulars.y() += M_PI;
    // U: roll counterclockwise (Rz-)
    if (ImGui::IsKeyDown(ImGuiKey_U))
        eulars.z() -= M_PI;
    // O: roll clockwise (Rz+)
    if (ImGui::IsKeyDown(ImGuiKey_O))
        eulars.z() += M_PI;
    // Angular Velocity
    eulars *= keyboard_anglular_velocity_;
    // To rotation matrix
    Eigen::AngleAxisf roll_angle(eulars.z(), Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yaw_angle(eulars.y(), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitch_angle(eulars.x(), Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = roll_angle * yaw_angle * pitch_angle;
    Eigen::Matrix3f rotating = q.matrix();
    //--------------

    //---Apply---
    Eigen::Matrix3f R = Twc_main_.rotationMatrix();
    Twc_main_.translation() += (R * translating);
    Twc_main_.setRotationMatrix(R * rotating);
}