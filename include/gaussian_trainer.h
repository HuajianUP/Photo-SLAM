/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#pragma once

#include <torch/torch.h>

#include <iomanip>
#include <random>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <vector>
#include <unordered_map>

#include <opencv2/opencv.hpp>

#include "ORB-SLAM3/include/System.h"

#include "loss_utils.h"
#include "gaussian_parameters.h"
#include "gaussian_model.h"
#include "gaussian_scene.h"
#include "gaussian_renderer.h"


class GaussianTrainer
{
public:
    GaussianTrainer();

    static void trainingOnce(
        std::shared_ptr<GaussianScene> scene,
        std::shared_ptr<GaussianModel> gaussians,
        GaussianModelParams& dataset,
        GaussianOptimizationParams& opt,
        GaussianPipelineParams& pipe,
        torch::DeviceType device_type = torch::kCUDA,
        std::vector<int> testing_iterations = {},
        std::vector<int> saving_iterations = {},
        std::vector<int> checkpoint_iterations = {}/*, checkpoint*/);

    static void trainingReport(
        int iteration,
        int num_iterations,
        torch::Tensor& Ll1,
        torch::Tensor& loss,
        float ema_loss_for_log,
        std::function<torch::Tensor(torch::Tensor&, torch::Tensor&)> l1_loss,
        int64_t elapsed_time,
        GaussianModel& gaussians,
        GaussianScene& scene,
        GaussianPipelineParams& pipe,
        torch::Tensor& background);

};
