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

#include "include/gaussian_mapper.h"

GaussianMapper::GaussianMapper(
    std::shared_ptr<ORB_SLAM3::System> pSLAM,
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,
    torch::DeviceType device_type)
    : pSLAM_(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    // Random seed
    std::srand(seed);
    torch::manual_seed(seed);

    // Device
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        model_params_.data_device_ = "cpu";
    }

    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

    std::vector<float> bg_color;
    if (model_params_.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};
    else
        bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    // Initialize scene and model
    gaussians_ = std::make_shared<GaussianModel>(model_params_);
    scene_ = std::make_shared<GaussianScene>(model_params_);

    // Mode
    if (!pSLAM) {
        // NO SLAM
        return;
    }

    // Sensors
    switch (pSLAM->getSensorType())
    {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
    {
        this->sensor_type_ = MONOCULAR;
    }
    break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
    {
        this->sensor_type_ = STEREO;
        this->stereo_baseline_length_ = pSLAM->getSettings()->b();
        this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
            this->stereo_min_disparity_,
            this->stereo_num_disparity_);
        this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
        stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);
    }
    break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD:
    {
        this->sensor_type_ = RGBD;
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    // Cameras
    // TODO: not only monocular
    auto settings = pSLAM->getSettings();
    cv::Size SLAM_im_size = settings->newImSize();
    UndistortParams undistort_params(
        SLAM_im_size,
        settings->camera1DistortionCoef()
    );

    auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
    for (auto& SLAM_camera : vpCameras) {
        Camera camera;
        camera.camera_id_ = SLAM_camera->GetId();
        if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
            camera.setModelId(Camera::CameraModelType::PINHOLE);
            float SLAM_fx = SLAM_camera->getParameter(0);
            float SLAM_fy = SLAM_camera->getParameter(1);
            float SLAM_cx = SLAM_camera->getParameter(2);
            float SLAM_cy = SLAM_camera->getParameter(3);

            // Old K, i.e. K in SLAM
            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << SLAM_fx, 0.f, SLAM_cx,
                        0.f, SLAM_fy, SLAM_cy,
                        0.f, 0.f, 1.f
            );

            // camera.width_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.width
            //                                              : graphics_utils::roundToIntegerMultipleOf16(
            //                                                    undistort_params.old_size_.width);
            camera.width_ = undistort_params.old_size_.width;
            float x_ratio = static_cast<float>(camera.width_) / undistort_params.old_size_.width;

            // camera.height_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.height
            //                                               : graphics_utils::roundToIntegerMultipleOf16(
            //                                                     undistort_params.old_size_.height);
            camera.height_ = undistort_params.old_size_.height;
            float y_ratio = static_cast<float>(camera.height_) / undistort_params.old_size_.height;

            camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
            camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
            camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
                camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
            }

            camera.params_[0]/*new fx*/= SLAM_fx * x_ratio;
            camera.params_[1]/*new fy*/= SLAM_fy * y_ratio;
            camera.params_[2]/*new cx*/= SLAM_cx * x_ratio;
            camera.params_[3]/*new cy*/= SLAM_cy * y_ratio;

            cv::Mat K_new = (
                cv::Mat_<float>(3, 3)
                    << camera.params_[0], 0.f, camera.params_[2],
                        0.f, camera.params_[1], camera.params_[3],
                        0.f, 0.f, 1.f
            );

            // Undistortion
            if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
                undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

            camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_sub_undistort_mask;
            int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
            int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
            cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                       cv::Size(viewer_image_width_, viewer_image_height_));
            viewer_sub_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_sub_undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

            if (this->sensor_type_ == STEREO) {
                camera.stereo_bf_ = stereo_baseline_length_ * camera.params_[0];
                if (this->stereo_Q_.cols != 4) {
                    this->stereo_Q_ = cv::Mat(4, 4, CV_32FC1);
                    this->stereo_Q_.setTo(0.0f);
                    this->stereo_Q_.at<float>(0, 0) = 1.0f;
                    this->stereo_Q_.at<float>(0, 3) = -camera.params_[2];
                    this->stereo_Q_.at<float>(1, 1) = 1.0f;
                    this->stereo_Q_.at<float>(1, 3) = -camera.params_[3];
                    this->stereo_Q_.at<float>(2, 3) = camera.params_[0];
                    this->stereo_Q_.at<float>(3, 2) = 1.0f / stereo_baseline_length_;
                }
            }
        }
        else if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_FISHEYE) {
            camera.setModelId(Camera::CameraModelType::FISHEYE);
        }
        else {
            camera.setModelId(Camera::CameraModelType::INVALID);
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }
}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }

    std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path << std::endl;
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // Model parameters
    model_params_.sh_degree_ =
        settings_file["Model.sh_degree"].operator int();
    model_params_.resolution_ =
        settings_file["Model.resolution"].operator float();
    model_params_.white_background_ =
        (settings_file["Model.white_background"].operator int()) != 0;
    model_params_.eval_ =
        (settings_file["Model.eval"].operator int()) != 0;

    // Pipeline Parameters
    z_near_ =
        settings_file["Camera.z_near"].operator float();
    z_far_ =
        settings_file["Camera.z_far"].operator float();

    monocular_inactive_geo_densify_max_pixel_dist_ =
        settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].operator float();
    stereo_min_disparity_ =
        settings_file["Stereo.min_disparity"].operator int();
    stereo_num_disparity_ =
        settings_file["Stereo.num_disparity"].operator int();
    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    min_num_initial_map_kfs_ = 
        static_cast<unsigned long>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ = 
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    local_BA_increased_times_of_use_ = 
        settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ = 
        settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    stable_num_iter_existence_ =
        settings_file["Mapper.stable_num_iter_existence"].operator int();

    pipe_params_.convert_SHs_ =
        (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ =
        (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    do_gaus_pyramid_training_ =
        (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }

    keyframe_record_interval_ = 
        settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ = 
        settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ = 
        (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_ = 
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ = 
        (settings_file["Record.record_loss_image"].operator int()) != 0;
    training_report_interval_ = 
        settings_file["Record.training_report_interval"].operator int();
    record_loop_ply_ =
        (settings_file["Record.record_loop_ply"].operator int()) != 0;

    // Optimization Parameters
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.position_lr_init_ =
        settings_file["Optimization.position_lr_init"].operator float();
    opt_params_.position_lr_final_ =
        settings_file["Optimization.position_lr_final"].operator float();
    opt_params_.position_lr_delay_mult_ =
        settings_file["Optimization.position_lr_delay_mult"].operator float();
    opt_params_.position_lr_max_steps_ =
        settings_file["Optimization.position_lr_max_steps"].operator int();
    opt_params_.feature_lr_ =
        settings_file["Optimization.feature_lr"].operator float();
    opt_params_.opacity_lr_ =
        settings_file["Optimization.opacity_lr"].operator float();
    opt_params_.scaling_lr_ =
        settings_file["Optimization.scaling_lr"].operator float();
    opt_params_.rotation_lr_ =
        settings_file["Optimization.rotation_lr"].operator float();

    opt_params_.percent_dense_ =
        settings_file["Optimization.percent_dense"].operator float();
    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();
    opt_params_.densification_interval_ =
        settings_file["Optimization.densification_interval"].operator int();
    opt_params_.opacity_reset_interval_ =
        settings_file["Optimization.opacity_reset_interval"].operator int();
    opt_params_.densify_from_iter_ =
        settings_file["Optimization.densify_from_iter_"].operator int();
    opt_params_.densify_until_iter_ =
        settings_file["Optimization.densify_until_iter"].operator int();
    opt_params_.densify_grad_threshold_ =
        settings_file["Optimization.densify_grad_threshold"].operator float();

    prune_big_point_after_iter_ =
        settings_file["Optimization.prune_big_point_after_iter"].operator int();
    densify_min_opacity_ =
        settings_file["Optimization.densify_min_opacity"].operator float();

    // Viewer Parameters
    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();
}

void GaussianMapper::run()
{
    // First loop: Initial gaussian mapping
    while (!isStopped()) {
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions()) {
            pSLAM_->getAtlas()->clearMappingOperation();

            // Get initial sparse map
            auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
            std::vector<ORB_SLAM3::MapPoint*> vpMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                vpKFs = pMap->GetAllKeyFrames();
                vpMPs = pMap->GetAllMapPoints();
                for (const auto& pMP : vpMPs){
                    Point3D point3D;
                    auto pos = pMP->GetWorldPos();
                    point3D.xyz_(0) = pos.x();
                    point3D.xyz_(1) = pos.y();
                    point3D.xyz_(2) = pos.z();
                    auto color = pMP->GetColorRGB();
                    point3D.color_(0) = color(0);
                    point3D.color_(1) = color(1);
                    point3D.color_(2) = color(2);
                    scene_->cachePoint3D(pMP->mnId, point3D);
                }
                for (const auto& pKF : vpKFs){
                    std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(pKF->mnId, getIteration());
                    new_kf->zfar_ = z_far_;
                    new_kf->znear_ = z_near_;
                    // Pose
                    auto pose = pKF->GetPose();
                    new_kf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;
                    try {
                        // Camera
                        Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                        new_kf->setCameraParams(camera);

                        // Image (left if STEREO)
                        cv::Mat imgRGB = pKF->imgLeftRGB;
                        if (this->sensor_type_ == STEREO)
                            imgRGB_undistorted = imgRGB;
                        else
                            camera.undistortImage(imgRGB, imgRGB_undistorted);
                        // Auxiliary Image
                        cv::Mat imgAux = pKF->imgAuxiliary;
                        if (this->sensor_type_ == RGBD)
                            camera.undistortImage(imgAux, imgAux_undistorted);
                        else
                            imgAux_undistorted = imgAux;

                        new_kf->original_image_ =
                            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                        new_kf->img_filename_ = pKF->mNameFile;
                        new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                        new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                        new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
                    }
                    catch (std::out_of_range) {
                        throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
                    }
                    new_kf->computeTransformTensors();
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);

                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    // Features
                    std::vector<float> pixels;
                    std::vector<float> pointsLocal;
                    pKF->GetKeypointInfo(pixels, pointsLocal);
                    new_kf->kps_pixel_ = std::move(pixels);
                    new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;
                }
            }

            // Prepare multi resolution images for training
            for (auto& kfit : scene_->keyframes()) {
                auto pkf = kfit.second;
                if (device_type_ == torch::kCUDA) {
                    cv::cuda::GpuMat img_gpu;
                    img_gpu.upload(pkf->img_undist_);
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::cuda::GpuMat img_resized;
                        cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::Mat img_resized;
                        cv::resize(pkf->img_undist_, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                    }
                }
            }

            // Prepare for training
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
                std::unique_lock<std::mutex> lock(mutex_settings_);
                gaussians_->trainingSetup(opt_params_);
            }

            // Invoke training once
            trainForOneIteration();

            // Finish initial mapping loop
            initial_mapped_ = true;
            break;
        }
        else if (pSLAM_->isShutDown()) {
            break;
        }
        else {
            // Initial conditions not satisfied
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Second loop: Incremental gaussian mapping
    int SLAM_stop_iter = 0;
    while (!isStopped()) {
        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations();
            if (cull_keyframes_)
                cullKeyframes();
        }

        // Invoke training once
        trainForOneIteration();

        if (pSLAM_->isShutDown()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;
    }

    // Third loop: Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}

void GaussianMapper::trainColmap()
{
    // Prepare multi resolution images for training
    for (auto& kfit : scene_->keyframes()) {
        auto pkf = kfit.second;
        increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());
        if (device_type_ == torch::kCUDA) {
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(pkf->img_undist_);
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::cuda::GpuMat img_resized;
                cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
            }
        }
        else {
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::Mat img_resized;
                cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
            }
        }
    }

    // Prepare for training
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
        gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
        std::unique_lock<std::mutex> lock(mutex_settings_);
        gaussians_->trainingSetup(opt_params_);
        this->initial_mapped_ = true;
    }

    // Main loop: gaussian splatting training
    while (!isStopped()) {
        // Invoke training once
        trainForOneIteration();

        if (getIteration() >= opt_params_.iterations_)
            break;
    }

    // Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}

/**
 * @brief The training iteration body
 * 
 */
void GaussianMapper::trainForOneIteration()
{
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    // Pick a random Camera
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }

    writeKeyframeUsedTimes(result_dir_ / "used_times");

    // if (isdoingInactiveGeoDensify() && !viewpoint_cam->done_inactive_geo_densify_)
    //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;
    if (isdoingGausPyramidTraining())
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_.cuda();
        mask = undistort_mask_[viewpoint_cam->camera_id_];
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];
    }

    // Mutex lock for usage of the gaussian model
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    // Every 1000 its we increase the levels of SH up to a maximum degree
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
        default_sh_ += 1;
    // if (isdoingGausPyramidTraining())
    //     gaussians_->setShDegree(training_level);
    // else
        gaussians_->setShDegree(default_sh_);

    // Update learning rate
    if (pSLAM_) {
        int used_times = kfs_used_times_[viewpoint_cam->fid_];
        int step = (used_times <= opt_params_.position_lr_max_steps_ ? used_times : opt_params_.position_lr_max_steps_);
        float position_lr = gaussians_->updateLearningRate(step);
        setPositionLearningRateInit(position_lr);
    }
    else {
        gaussians_->updateLearningRate(getIteration());
    }

    gaussians_->setFeatureLearningRate(featureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());

    // Render
    auto render_pkg = GaussianRenderer::render(
        viewpoint_cam,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    auto rendered_image = std::get<0>(render_pkg);
    auto viewspace_point_tensor = std::get<1>(render_pkg);
    auto visibility_filter = std::get<2>(render_pkg);
    auto radii = std::get<3>(render_pkg);

    // Get rid of black edges caused by undistortion
    torch::Tensor masked_image = rendered_image * mask;

    // Loss
    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    float lambda_dssim = lambdaDssim();
    auto loss = (1.0 - lambda_dssim) * Ll1
                + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_));
    loss.backward();

    torch::cuda::synchronize();

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(masked_image, gt_image, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_);

        // Densification
        if (getIteration() < opt_params_.densify_until_iter_) {
            // Keep track of max radii in image-space for pruning
            gaussians_->max_radii2D_.index_put_(
                {visibility_filter},
                torch::max(gaussians_->max_radii2D_.index({visibility_filter}),
                            radii.index({visibility_filter})));
            // if (!isdoingGausPyramidTraining() || training_level < num_gaus_pyramid_sub_levels_)
                gaussians_->addDensificationStats(viewspace_point_tensor, visibility_filter);

            if ((getIteration() > opt_params_.densify_from_iter_) &&
                (getIteration() % densifyInterval()== 0)) {
                int size_threshold = (getIteration() > prune_big_point_after_iter_) ? 20 : 0;
                gaussians_->densifyAndPrune(
                    densifyGradThreshold(),
                    densify_min_opacity_,//0.005,//
                    scene_->cameras_extent_,
                    size_threshold
                );
            }

            if (opacityResetInterval()
                && (getIteration() % opacityResetInterval() == 0
                    ||(model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)))
                gaussians_->resetOpacity();
        }

        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

        // Log and save
        if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
            GaussianTrainer::trainingReport(
                getIteration(),
                opt_params_.iterations_,
                Ll1,
                loss,
                ema_loss_for_log_,
                loss_utils::l1_loss,
                iter_time,
                *gaussians_,
                *scene_,
                pipe_params_,
                background_
            );
        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            // || loop_closure_iteration_
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(getIteration()) / "ply");
        }

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Optimizer step
        if (getIteration() < opt_params_.iterations_) {
            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);
        }
    }
}

bool GaussianMapper::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void GaussianMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

bool GaussianMapper::hasMetInitialMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}

bool GaussianMapper::hasMetIncrementalMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}

void GaussianMapper::combineMappingOperations()
{
    // Get Mapping Operations
    while (pSLAM_->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            pSLAM_->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            // std::cout << "[Gaussian Mapper]Local BA Detected."
            //           << std::endl;

            // Get new keyframes
            auto& associated_kfs = opr.associatedKeyFrames();

            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
                // If the keyframe is already in the scene, only update the pose.
                // Otherwise create a new one
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();

                    // Give local BA keyframes times of use
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                }
                else {
                    handleNewKeyframe(kf);
                }
            }

            // Get new points
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            std::cout << "[Gaussian Mapper]Loop Closure Detected."
                      << std::endl;

            // Get the loop keyframe scale modification factor
            float loop_kf_scale = opr.mfScale;

            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
            auto& associated_kfs = opr.associatedKeyFrames();
            // Mark the transformed points to avoid transforming more than once
            torch::Tensor point_not_transformed_flags =
                torch::full(
                    {gaussians_->xyz_.size(0)},
                    true,
                    torch::TensorOptions().device(device_type_).dtype(torch::kBool));
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
            int num_transformed = 0;
            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
                // In case new points are added in handleNewKeyframe()
                int64_t num_new_points = gaussians_->xyz_.size(0) - point_not_transformed_flags.size(0);
                if (num_new_points > 0)
                    point_not_transformed_flags = torch::cat({
                        point_not_transformed_flags,
                        torch::full({num_new_points}, true, point_not_transformed_flags.options())},
                        /*dim=*/0);
                // If kf is already in the scene, evaluate the change in pose,
                // if too large we perform loop correction on its visible model points.
                // If not in the scene, create a new one.
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    // If is loop closure kf
// if (std::get<4>(kf)) {
// renderAndRecordKeyframe(pkf, result_dir_, "_0_before_loop_correction");
                        Sophus::SE3f original_pose = pkf->getPosef(); // original_pose = old, inv_pose = new
                        Sophus::SE3f inv_pose = pose.inverse();
                        Sophus::SE3f diff_pose = inv_pose * original_pose;
                        bool large_rot = !diff_pose.rotationMatrix().isApprox(
                            Eigen::Matrix3f::Identity(), large_rot_th_);
                        bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                            1.0, large_trans_th_);
                        if (large_rot || large_trans) {
                            std::cout << "[Gaussian Mapper]Large loop correction detected, transforming visible points of kf "
                                    << kfid << std::endl;
                            diff_pose.translation() -= inv_pose.translation(); // t = (R_new * t_old + t_new) - t_new
                            diff_pose.translation() *= loop_kf_scale;          // t = s * (R_new * t_old)
                            diff_pose.translation() += inv_pose.translation(); // t = (s * R_new * t_old) + t_new
                            torch::Tensor diff_pose_tensor =
                                tensor_utils::EigenMatrix2TorchTensor(
                                    diff_pose.matrix(), device_type_).transpose(0, 1);
                            {
                                std::unique_lock<std::mutex> lock_render(mutex_render_);
                                gaussians_->scaledTransformVisiblePointsOfKeyframe(
                                    point_not_transformed_flags,
                                    diff_pose_tensor,
                                    pkf->world_view_transform_,
                                    pkf->full_proj_transform_,
                                    pkf->creation_iter_,
                                    stableNumIterExistence(),
                                    num_transformed,
                                    loop_kf_scale); // selected xyz *= s
                            }
                            // Give loop keyframes times of use
                            increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
// renderAndRecordKeyframe(pkf, result_dir_, "_1_after_loop_transforming_points");
// std::cout<<num_transformed<<std::endl;
                        }
// }
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();
// if (std::get<4>(kf)) renderAndRecordKeyframe(pkf, result_dir_, "_2_after_pose_correction");
                }
                else {
                    handleNewKeyframe(kf);
                }
            }
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
// keyframesToJson(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));

            // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }

            // Mark this iteration
            loop_closure_iteration_ = true;
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
        {
            std::cout << "[Gaussian Mapper]Scale refinement Detected. Transforming all kfs and points..."
                      << std::endl;

            float s = opr.mfScale;
            Sophus::SE3f& T = opr.mT;
            if (initial_mapped_) {
                // Apply the scaled transformation on gaussian model points
                {
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    gaussians_->applyScaledTransformation(s, T);
                }
                // Apply the scaled transformation to the scene
                scene_->applyScaledTransformation(s, T);
            }
            else { // TODO: the workflow should not come here, delete this branch
                // Apply the scaled transformation to the cached points
                for (auto& pt : scene_->cached_point_cloud_) {
                    // pt <- (s * Ryw * pt + tyw)
                    auto& pt_xyz = pt.second.xyz_;
                    pt_xyz *= s;
                    pt_xyz = T.cast<double>() * pt_xyz;
                }

                // Apply the scaled transformation on gaussian keyframes
                for (auto& kfit : scene_->keyframes()) {
                    std::shared_ptr<GaussianKeyframe> pkf = kfit.second;
                    Sophus::SE3f Twc = pkf->getPosef().inverse();
                    Twc.translation() *= s;
                    Sophus::SE3f Tyc = T * Twc;
                    Sophus::SE3f Tcy = Tyc.inverse();
                    pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                    pkf->computeTransformTensors();
                }
            }
        }
        break;

        default:
        {
            throw std::runtime_error("MappingOperation type not supported!");
        }
        break;
        }
    }
}

void GaussianMapper::handleNewKeyframe(
    std::tuple< unsigned long/*Id*/,
                unsigned long/*CameraId*/,
                Sophus::SE3f/*pose*/,
                cv::Mat/*image*/,
                bool/*isLoopClosure*/,
                cv::Mat/*auxiliaryImage*/,
                std::vector<float>,
                std::vector<float>,
                std::string> &kf)
{
    std::shared_ptr<GaussianKeyframe> pkf =
        std::make_shared<GaussianKeyframe>(std::get<0>(kf), getIteration());
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>());
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(std::get<1>(kf));
        pkf->setCameraParams(camera);

        // Image (left if STEREO)
        cv::Mat imgRGB = std::get<3>(kf);
        if (this->sensor_type_ == STEREO)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
        // Auxiliary Image
        cv::Mat imgAux = std::get<5>(kf);
        if (this->sensor_type_ == RGBD)
            camera.undistortImage(imgAux, imgAux_undistorted);
        else
            imgAux_undistorted = imgAux;

        pkf->original_image_ =
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
        pkf->img_filename_ = std::get<8>(kf);
        pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
    }
    // Add the new keyframe to the scene
    pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;
    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);

    // Prepare multi resolution images for training
    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat img_gpu;
        img_gpu.upload(pkf->img_undist_);
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }
}

void GaussianMapper::generateKfidRandomShuffle()
{
// if (viewpoint_sliding_window_.empty())
//     return;

// std::size_t sliding_window_size = viewpoint_sliding_window_.size();
// kfid_shuffle_.resize(sliding_window_size);
// std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
// std::mt19937 g(rd_());
// std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    if (scene_->keyframes().empty())
        return;

    std::size_t nkfs = scene_->keyframes().size();
    kfid_shuffle_.resize(nkfs);
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
    std::mt19937 g(rd_());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe()
{
// auto t1 = std::chrono::steady_clock::now();
    if (scene_->keyframes().empty())
        return nullptr;

    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;
    int random_cam_idx;

    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do {
            // Next shuffled idx
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            // Add 1 time of use to all kfs if they are all unavalible
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];
    
    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);

// auto t2 = std::chrono::steady_clock::now();
// auto t21 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
// std::cout<<t21 <<" ns"<<std::endl;
    return viewpoint_cam;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    // Get randomly
    int nkfs = static_cast<int>(scene_->keyframes().size());
    int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
    auto random_cam_it = scene_->keyframes().begin();
    for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    return viewpoint_cam;
}

void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,
    int times)
{
    pkf->remaining_times_of_use_ += times;
}

void GaussianMapper::cullKeyframes()
{
    std::unordered_set<unsigned long> kfids =
        pSLAM_->getAtlas()->GetCurrentKeyFrameIds();
    std::vector<unsigned long> kfids_to_erase;
    std::size_t nkfs = scene_->keyframes().size();
    kfids_to_erase.reserve(nkfs);
    for (auto& kfit : scene_->keyframes()) {
        unsigned long kfid = kfit.first;
        if (kfids.find(kfid) == kfids.end()) {
            kfids_to_erase.emplace_back(kfid);
        }
    }

    for (auto& kfid : kfids_to_erase) {
        scene_->keyframes().erase(kfid);
    }
}

void GaussianMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<GaussianKeyframe> pkf)
{
// auto start_timing = std::chrono::steady_clock::now();
    torch::NoGradGuard no_grad;

    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        assert(pkf->kps_pixel_.size() % 2 == 0);
        int N = pkf->kps_pixel_.size() / 2;
        torch::Tensor kps_pixel_tensor = torch::from_blob(
            pkf->kps_pixel_.data(), {N, 2},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_point_local_tensor = torch::from_blob(
            pkf->kps_point_local_.data(), {N, 3},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_has3D_tensor = torch::where(
            kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f, true, false);

        cv::cuda::GpuMat rgb_gpu;
        rgb_gpu.upload(pkf->img_undist_);
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        auto result =
            monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                kps_pixel_tensor, kps_has3D_tensor, kps_point_local_tensor, colors,
                monocular_inactive_geo_densify_max_pixel_dist_, pkf->intr_, pkf->image_width_);
        torch::Tensor& points3D_valid = std::get<0>(result);
        torch::Tensor& colors_valid = std::get<1>(result);
        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);
        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case STEREO:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
        cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

        rgb_left_gpu.upload(pkf->img_undist_);
        rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

        // From CV_32FC3 to CV_32FC1
        cv::cuda::cvtColor(rgb_left_gpu, gray_left_gpu, cv::COLOR_RGB2GRAY);
        cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

        // From CV_32FC1 to CV_8UC1
        gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);
        gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

        // Compute disparity
        cv::cuda::GpuMat cv_disp;
        stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
        cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

        // Reproject to get 3D points
        cv::cuda::GpuMat cv_points3D;
        cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
        disp = disp.flatten(0, 1).contiguous();
        torch::Tensor points3D = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();
    
        // Clear undisired and unreliable stereo points
        torch::Tensor point_valid_flags = torch::full(
            {disp.size(0)}, false, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            // int u = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]);
            // if (u < 0.3 * width || u > 0.7 * width)
            point_valid_flags[idx] = true;
            // idx += width;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx - 3] = true;
            //     point_valid_flags[idx - 2] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx] = true;
            // }
            // idx -= (2 * width);
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            //     point_valid_flags[idx + 3] = true;
            // }
            // idx += width;
            // idx += 3;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx - 2] = true;
            // }
            // idx -= 6;
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            // }
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()), true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()), true, false));

        torch::Tensor points3D_valid = points3D.index({point_valid_flags});
        torch::Tensor colors_valid = colors.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case RGBD:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
        img_rgb_gpu.upload(pkf->img_undist_);
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor depth = tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        // To clear undisired and unreliable depth
        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)}, false/*true*/, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth > RGBD_min_depth_, true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth < RGBD_max_depth_, true, false));

        torch::Tensor colors_valid = rgb.index({point_valid_flags});

        // Reproject to get 3D points
        torch::Tensor points3D_valid;
        Camera& camera = scene_->cameras_.at(pkf->camera_id_);
        switch (camera.model_id_)
        {
        case Camera::PINHOLE:
        {
            points3D_valid = reprojectDepthPinhole(
                depth, point_valid_flags, pkf->intr_, pkf->image_width_);
        }
        break;
        case Camera::FISHEYE:
        {
            //TODO: support fisheye camera?
            throw std::runtime_error("[Gaussian Mapper]Fisheye cameras are not supported currently!");
        }
        break;
        default:
        {
            throw std::runtime_error("[Gaussian Mapper]Invalid camera model!");
        }
        break;
        }
        points3D_valid = points3D_valid.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    pkf->done_inactive_geo_densify_ = true;
    ++depth_cached_;

    if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;
        // Add new points to the model
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        gaussians_->increasePcd(depth_cache_points_, depth_cache_colors_, getIteration());
    }

// auto end_timing = std::chrono::steady_clock::now();
// auto completion_time = std::chrono::duration_cast<std::chrono::milliseconds>(
//                 end_timing - start_timing).count();
// std::cout << "[Gaussian Mapper]increasePcdByKeyframeInactiveGeoDensify() takes "
//             << completion_time
//             << " ms"
//             << std::endl;
}

// bool GaussianMapper::needInterruptTraining()
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     return this->interrupt_training_;
// }

// void GaussianMapper::setInterruptTraining(const bool interrupt_training)
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     this->interrupt_training_ = interrupt_training;
// }

void GaussianMapper::recordKeyframeRendered(
        torch::Tensor &rendered,
        torch::Tensor &ground_truth,
        unsigned long kfid,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::string name_suffix)
{
    if (record_rendered_image_) {
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
        cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
    }

    if (record_ground_truth_image_) {
        auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
        cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
        gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
    }

    if (record_loss_image_) {
        torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
        auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
        cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
        loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
    }
}

cv::Mat GaussianMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRenderer::render(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }

    // Result
    torch::Tensor masked_image;
    if (main_vision)
        masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
    else
        masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}

void GaussianMapper::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframe> pkf,
    float &dssim,
    float &psnr,
    float &psnr_gs,
    double &render_time,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::string name_suffix)
{
    auto start_timing = std::chrono::steady_clock::now();
    auto render_pkg = GaussianRenderer::render(
        pkf,
        pkf->image_height_,
        pkf->image_width_,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    auto rendered_image = std::get<0>(render_pkg);
    torch::Tensor masked_image = rendered_image * undistort_mask_[pkf->camera_id_];
    torch::cuda::synchronize();
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_time = 1e-6 * render_time_ns;
    auto gt_image = pkf->original_image_;

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();
    psnr_gs = loss_utils::psnr_gaussian_splatting(masked_image, gt_image).item().toFloat();

    recordKeyframeRendered(masked_image, gt_image, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, name_suffix);    
}

void GaussianMapper::renderAndRecordAllKeyframes(
    std::string name_suffix)
{
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
    }

    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Gaussian Mapper]Render time statistics: keyframe id, time(milliseconds)" << std::endl;

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Gaussian Mapper]keyframe id, psnr" << std::endl;

    std::filesystem::path psnr_gs_path = result_dir / "psnr_gaussian_splatting.txt";
    std::ofstream out_psnr_gs(psnr_gs_path);
    out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting" << std::endl;

    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr, psnr_gs;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        renderAndRecordKeyframe((*kfit).second, dssim, psnr, psnr_gs, render_time, image_dir, image_gt_dir, image_loss_dir);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;
        out_psnr_gs << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

        ++kfit;
    }
}

void GaussianMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    keyframesToJson(result_dir);
    saveModelParams(result_dir);

    std::filesystem::path ply_dir = result_dir / "point_cloud";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    gaussians_->savePly(ply_dir / "point_cloud.ply");
    gaussians_->saveSparsePointsPly(result_dir / "input.ply");
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path result_path = result_dir / "cameras.json";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + result_path.string());

    Json::Value json_root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    int i = 0;
    for (const auto& kfit : scene_->keyframes()) {
        const auto pkf = kfit.second;
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = pkf->t_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f Twc = Rt.inverse();
        Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
        Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

        Json::Value json_kf;
        json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
        json_kf["img_name"] = pkf->img_filename_; //(std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_));
        json_kf["width"] = pkf->image_width_;
        json_kf["height"] = pkf->image_height_;

        json_kf["position"][0] = pos.x();
        json_kf["position"][1] = pos.y();
        json_kf["position"][2] = pos.z();

        json_kf["rotation"][0][0] = rot(0, 0);
        json_kf["rotation"][0][1] = rot(0, 1);
        json_kf["rotation"][0][2] = rot(0, 2);
        json_kf["rotation"][1][0] = rot(1, 0);
        json_kf["rotation"][1][1] = rot(1, 1);
        json_kf["rotation"][1][2] = rot(1, 2);
        json_kf["rotation"][2][0] = rot(2, 0);
        json_kf["rotation"][2][1] = rot(2, 1);
        json_kf["rotation"][2][2] = rot(2, 2);

        json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
        json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

        json_root[i] = Json::Value(json_kf);
        ++i;
    }

    writer->write(json_root, &out_stream);
}

void GaussianMapper::saveModelParams(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / "cfg_args";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open file at " + result_path.string());

    out_stream << "Namespace("
               << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
               << "images=" << "\'" << model_params_.images_ << "\', "
               << "model_path=" << "\'" << model_params_.model_path_.string() << "\', "
               << "resolution=" << model_params_.resolution_ << ", "
               << "sh_degree=" << model_params_.sh_degree_ << ", "
               << "source_path=" << "\'" << model_params_.source_path_.string() << "\', "
               << "white_background=" << (model_params_.white_background_ ? "True" : "False") << ", "
               << ")";

    out_stream.close();
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Gaussian Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}

int GaussianMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void GaussianMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.position_lr_init_;
}
float GaussianMapper::featureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.feature_lr_;
}
float GaussianMapper::opacityLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_lr_;
}
float GaussianMapper::scalingLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.scaling_lr_;
}
float GaussianMapper::rotationLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.rotation_lr_;
}
float GaussianMapper::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}
int GaussianMapper::opacityResetInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_reset_interval_;
}
float GaussianMapper::densifyGradThreshold()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densify_grad_threshold_;
}
int GaussianMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densification_interval_;
}
int GaussianMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

void GaussianMapper::setPositionLearningRateInit(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = lr;
}
void GaussianMapper::setFeatureLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.feature_lr_ = lr;
}
void GaussianMapper::setOpacityLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_lr_ = lr;
}
void GaussianMapper::setScalingLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.scaling_lr_ = lr;
}
void GaussianMapper::setRotationLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.rotation_lr_ = lr;
}
void GaussianMapper::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.percent_dense_ = percent_dense;
    gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapper::setOpacityResetInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapper::setDensifyGradThreshold(const float th)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapper::setDensifyInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densification_interval_ = interval;
}
void GaussianMapper::setNewKeyframeTimesOfUse(const int times)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    new_keyframe_times_of_use_ = times;
}
void GaussianMapper::setStableNumIterExistence(const int niter)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    stable_num_iter_existence_ = niter;
}
void GaussianMapper::setKeepTraining(const bool keep)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    keep_training_ = keep;
}
void GaussianMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapper::setDoInactiveGeoDensify(const bool inactive_geo_densify)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    inactive_geo_densify_ = inactive_geo_densify;
}

VariableParameters GaussianMapper::getVaribleParameters()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    VariableParameters params;
    params.position_lr_init = opt_params_.position_lr_init_;
    params.feature_lr = opt_params_.feature_lr_;
    params.opacity_lr = opt_params_.opacity_lr_;
    params.scaling_lr = opt_params_.scaling_lr_;
    params.rotation_lr = opt_params_.rotation_lr_;
    params.percent_dense = opt_params_.percent_dense_;
    params.lambda_dssim = opt_params_.lambda_dssim_;
    params.opacity_reset_interval = opt_params_.opacity_reset_interval_;
    params.densify_grad_th = opt_params_.densify_grad_threshold_;
    params.densify_interval = opt_params_.densification_interval_;
    params.new_kf_times_of_use = new_keyframe_times_of_use_;
    params.stable_num_iter_existence = stable_num_iter_existence_;
    params.keep_training = keep_training_;
    params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
    params.do_inactive_geo_densify = inactive_geo_densify_;
    return params;
}

void GaussianMapper::setVaribleParameters(const VariableParameters &params)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = params.position_lr_init;
    opt_params_.feature_lr_ = params.feature_lr;
    opt_params_.opacity_lr_ = params.opacity_lr;
    opt_params_.scaling_lr_ = params.scaling_lr;
    opt_params_.rotation_lr_ = params.rotation_lr;
    opt_params_.percent_dense_ = params.percent_dense;
    gaussians_->setPercentDense(params.percent_dense);
    opt_params_.lambda_dssim_ = params.lambda_dssim;
    opt_params_.opacity_reset_interval_ = params.opacity_reset_interval;
    opt_params_.densify_grad_threshold_ = params.densify_grad_th;
    opt_params_.densification_interval_ = params.densify_interval;
    new_keyframe_times_of_use_ = params.new_kf_times_of_use;
    stable_num_iter_existence_ = params.stable_num_iter_existence;
    keep_training_ = params.keep_training;
    do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
    inactive_geo_densify_ = params.do_inactive_geo_densify;
}

void GaussianMapper::loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path)
{
    this->gaussians_->loadPly(ply_path);

    // Camera
    if (!camera_path.empty() && std::filesystem::exists(camera_path)) {
        cv::FileStorage camera_file(camera_path.string().c_str(), cv::FileStorage::READ);
        if(!camera_file.isOpened())
            throw std::runtime_error("[Gaussian Mapper]Failed to open settings file at: " + camera_path.string());

        Camera camera;
        camera.camera_id_ = 0;
        camera.width_ = camera_file["Camera.w"].operator int();
        camera.height_ = camera_file["Camera.h"].operator int();

        std::string camera_type = camera_file["Camera.type"].string();
        if (camera_type == "Pinhole") {
            camera.setModelId(Camera::CameraModelType::PINHOLE);

            float fx = camera_file["Camera.fx"].operator float();
            float fy = camera_file["Camera.fy"].operator float();
            float cx = camera_file["Camera.cx"].operator float();
            float cy = camera_file["Camera.cy"].operator float();

            float k1 = camera_file["Camera.k1"].operator float();
            float k2 = camera_file["Camera.k2"].operator float();
            float p1 = camera_file["Camera.p1"].operator float();
            float p2 = camera_file["Camera.p2"].operator float();
            float k3 = camera_file["Camera.k3"].operator float();

            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << fx, 0.f, cx,
                        0.f, fy, cy,
                        0.f, 0.f, 1.f
            );

            camera.params_[0] = fx;
            camera.params_[1] = fy;
            camera.params_[2] = cx;
            camera.params_[3] = cy;

            std::vector<float> dist_coeff = {k1, k2, p1, p2, k3};
            camera.dist_coeff_ = cv::Mat(5, 1, CV_32F, dist_coeff.data());
            camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, false);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

        }
        else {
            throw std::runtime_error("[Gaussian Mapper]Unsupported camera model: " + camera_path.string());
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }

    // Ready
    this->initial_mapped_ = true;
    increaseIteration();
}