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

#include "include/gaussian_parameters.h"

GaussianModelParams::GaussianModelParams(
    std::filesystem::path source_path,
    std::filesystem::path model_path,
    std::filesystem::path exec_path,
    int sh_degree,
    std::string images,
    float resolution,
    bool white_background,
    std::string data_device,
    bool eval)
    : sh_degree_(sh_degree),
      images_(images),
      resolution_(resolution),
      white_background_(white_background),
      data_device_(data_device),
      eval_(eval)
{
    if (source_path.is_absolute())
        source_path_ = source_path;
    else
        source_path_ = exec_path / source_path;

    if (model_path.is_absolute())
        model_path_ = model_path;
    else
        model_path_ = exec_path / model_path;
}

GaussianPipelineParams::GaussianPipelineParams(bool convert_SHs, bool compute_cov3D)
    : convert_SHs_(convert_SHs), compute_cov3D_(compute_cov3D)
{}

GaussianOptimizationParams::GaussianOptimizationParams(
    int iterations,
    float position_lr_init,
    float position_lr_final,
    float position_lr_delay_mult,
    int position_lr_max_steps,
    float feature_lr,
    float opacity_lr,
    float scaling_lr,
    float rotation_lr,
    float percent_dense,
    float lambda_dssim,
    int densification_interval,
    int opacity_reset_interval,
    int densify_from_iter,
    int densify_until_iter,
    float densify_grad_threshold)
    : iterations_(iterations),
      position_lr_init_(position_lr_init),
      position_lr_final_(position_lr_final),
      position_lr_delay_mult_(position_lr_delay_mult),
      position_lr_max_steps_(position_lr_max_steps),
      feature_lr_(feature_lr),
      opacity_lr_(opacity_lr_),
      scaling_lr_(scaling_lr),
      rotation_lr_(rotation_lr),
      percent_dense_(percent_dense),
      lambda_dssim_(lambda_dssim),
      densification_interval_(densification_interval),
      opacity_reset_interval_(opacity_reset_interval),
      densify_from_iter_(densify_from_iter),
      densify_until_iter_(densify_until_iter),
      densify_grad_threshold_(densify_grad_threshold)
{}