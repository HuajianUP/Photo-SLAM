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

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <map>
#include <random>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>

#include <jsoncpp/json/json.h>

#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "operate_points.h"
#include "stereo_vision.h"
#include "tensor_utils.h"
#include "gaussian_keyframe.h"
#include "gaussian_scene.h"
#include "gaussian_trainer.h"

#define CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(dir)                                       \
    if (!dir.empty() && !std::filesystem::exists(dir))                                      \
        if (!std::filesystem::create_directories(dir))                                      \
            throw std::runtime_error("Cannot create result directory at " + dir.string());
struct UndistortParams
{
    UndistortParams(
        const cv::Size& old_size,
        cv::Mat dist_coeff = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, 0.0f, 0.0f))
        : old_size_(old_size)
    {
        dist_coeff.copyTo(dist_coeff_);
    }

    cv::Size old_size_;
    cv::Mat dist_coeff_;
};


enum SystemSensorType
{
    INVALID = 0,
    MONOCULAR = 1,
    STEREO = 2,
    RGBD = 3
};

struct VariableParameters
{
    float position_lr_init;
    float feature_lr;
    float opacity_lr;
    float scaling_lr;
    float rotation_lr;
    float percent_dense;
    float lambda_dssim;
    int opacity_reset_interval;
    float densify_grad_th;
    int densify_interval;
    int new_kf_times_of_use;
    int stable_num_iter_existence; ///< loop closure correction

    bool keep_training;
    bool do_gaus_pyramid_training;
    bool do_inactive_geo_densify;
};

class GaussianMapper
{
public:
    GaussianMapper(
        std::shared_ptr<ORB_SLAM3::System> pSLAM,
        std::filesystem::path gaussian_config_file_path,
        std::filesystem::path result_dir = "",
        int seed = 0,
        torch::DeviceType device_type = torch::kCUDA);

    void readConfigFromFile(std::filesystem::path cfg_path);

    void run();
    void trainColmap();
    void trainForOneIteration();

    bool isStopped();
    void signalStop(const bool going_to_stop = true);

    cv::Mat renderFromPose(
        const Sophus::SE3f &Tcw,
        const int width,
        const int height,
        const bool main_vision = false);

    int getIteration();
    void increaseIteration(const int inc = 1);

    float positionLearningRateInit();
    float featureLearningRate();
    float opacityLearningRate();
    float scalingLearningRate();
    float rotationLearningRate();
    float percentDense();
    float lambdaDssim();
    int opacityResetInterval();
    float densifyGradThreshold();
    int densifyInterval();
    int newKeyframeTimesOfUse();
    int stableNumIterExistence();
    bool isKeepingTraining();
    bool isdoingGausPyramidTraining();
    bool isdoingInactiveGeoDensify();

    void setPositionLearningRateInit(const float lr);
    void setFeatureLearningRate(const float lr);
    void setOpacityLearningRate(const float lr);
    void setScalingLearningRate(const float lr);
    void setRotationLearningRate(const float lr);
    void setPercentDense(const float percent_dense);
    void setLambdaDssim(const float lambda_dssim);
    void setOpacityResetInterval(const int interval);
    void setDensifyGradThreshold(const float th);
    void setDensifyInterval(const int interval);
    void setNewKeyframeTimesOfUse(const int times);
    void setStableNumIterExistence(const int niter);
    void setKeepTraining(const bool keep);
    void setDoGausPyramidTraining(const bool gaus_pyramid);
    void setDoInactiveGeoDensify(const bool inactive_geo_densify);

    VariableParameters getVaribleParameters();
    void setVaribleParameters(const VariableParameters &params);

    GaussianModelParams& getGaussianModelParams() { return this->model_params_; }
    void setColmapDataPath(std::filesystem::path colmap_path) { this->model_params_.source_path_ = colmap_path; }
    void setSensorType(SystemSensorType sensor_type) { this->sensor_type_ = sensor_type; }

    void loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path = "");

protected:
    bool hasMetInitialMappingConditions();
    bool hasMetIncrementalMappingConditions();

    void combineMappingOperations();

    void handleNewKeyframe(std::tuple<unsigned long,
                                      unsigned long,
                                      Sophus::SE3f,
                                      cv::Mat,
                                      bool,
                                      cv::Mat,
                                      std::vector<float>,
                                      std::vector<float>,
                                      std::string> &kf);
    void generateKfidRandomShuffle();
    std::shared_ptr<GaussianKeyframe> useOneRandomSlidingWindowKeyframe();
    std::shared_ptr<GaussianKeyframe> useOneRandomKeyframe();
    void increaseKeyframeTimesOfUse(std::shared_ptr<GaussianKeyframe> pkf, int times);
    void cullKeyframes();

    void increasePcdByKeyframeInactiveGeoDensify(
        std::shared_ptr<GaussianKeyframe> pkf);

    // bool needInterruptTraining();
    // void setInterruptTraining(const bool interrupt_training);

    void recordKeyframeRendered(
        torch::Tensor &rendered,
        torch::Tensor &ground_truth,
        unsigned long kfid,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::string name_suffix = "");
    void renderAndRecordKeyframe(
        std::shared_ptr<GaussianKeyframe> pkf,
        float &dssim,
        float &psnr,
        float &psnr_gs,
        double &render_time,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::string name_suffix = "");
    void renderAndRecordAllKeyframes(
        std::string name_suffix = "");

    void savePly(std::filesystem::path result_dir);
    void keyframesToJson(std::filesystem::path result_dir);
    void saveModelParams(std::filesystem::path result_dir);
    void writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix = "");

public:
    // Parameters
    std::filesystem::path config_file_path_;

    // Model
    std::shared_ptr<GaussianModel> gaussians_;
    std::shared_ptr<GaussianScene> scene_;

    // SLAM system
    std::shared_ptr<ORB_SLAM3::System> pSLAM_;

    // Settings
    torch::DeviceType device_type_;
    int num_gaus_pyramid_sub_levels_ = 0;
    std::vector<int> kf_gaus_pyramid_times_of_use_;
    std::vector<float> kf_gaus_pyramid_factors_;

    bool viewer_camera_id_set_ = false;
    std::uint32_t viewer_camera_id_ = 0;
    float rendered_image_viewer_scale_ = 1.0f;
    float rendered_image_viewer_scale_main_ = 1.0f;

    float z_near_ = 0.01f;
    float z_far_ = 100.0f;

    // Data
    bool kfid_shuffled_ = false;
    std::map<camera_id_t, torch::Tensor> undistort_mask_;
    std::map<camera_id_t, torch::Tensor> viewer_main_undistort_mask_;
    std::map<camera_id_t, torch::Tensor> viewer_sub_undistort_mask_;

protected:
    // Parameters
    GaussianModelParams model_params_;
    GaussianOptimizationParams opt_params_;
    GaussianPipelineParams pipe_params_;

    // Data
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> viewpoint_sliding_window_;
    std::vector<std::size_t> kfid_shuffle_;
    std::size_t kfid_shuffle_idx_ = 0;
    std::map<std::size_t, int> kfs_used_times_;

    // Status
    bool initial_mapped_;
    bool interrupt_training_;
    bool stopped_;
    int iteration_;
    float ema_loss_for_log_;
    bool SLAM_ended_;
    bool loop_closure_iteration_;
    bool keep_training_ = false;
    int default_sh_ = 0;

    // Settings
    SystemSensorType sensor_type_;

    float monocular_inactive_geo_densify_max_pixel_dist_ = 20.0;
    float stereo_baseline_length_ = 0.0f;
    int stereo_min_disparity_ = 0;
    int stereo_num_disparity_ = 128;
    cv::Mat stereo_Q_;
    cv::Ptr<cv::cuda::StereoSGM> stereo_cv_sgm_;
    float RGBD_min_depth_ = 0.0f;
    float RGBD_max_depth_ = 100.0f;

    bool inactive_geo_densify_ = true;
    int depth_cached_ = 0;
    int max_depth_cached_ = 1;
    torch::Tensor depth_cache_points_;
    torch::Tensor depth_cache_colors_;

    unsigned long min_num_initial_map_kfs_;
    torch::Tensor background_;
    float large_rot_th_;
    float large_trans_th_;
    torch::Tensor override_color_;

    int new_keyframe_times_of_use_;
    int local_BA_increased_times_of_use_;
    int loop_closure_increased_times_of_use_;

    bool cull_keyframes_;
    int stable_num_iter_existence_;

    bool do_gaus_pyramid_training_;

    std::filesystem::path result_dir_;
    int keyframe_record_interval_;
    int all_keyframes_record_interval_;
    bool record_rendered_image_;
    bool record_ground_truth_image_;
    bool record_loss_image_;

    int training_report_interval_;
    bool record_loop_ply_;

    int prune_big_point_after_iter_;
    float densify_min_opacity_ = 20;

    // Tools
    std::random_device rd_;

    // Mutex
    std::mutex mutex_status_;
    std::mutex mutex_settings_;
    std::mutex mutex_render_; ///< the model is suppose to be read-only from outside
};