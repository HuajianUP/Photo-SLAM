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

#include "include/gaussian_model.h"

GaussianModel::GaussianModel(const int sh_degree)
    : active_sh_degree_(0), spatial_lr_scale_(0.0),
      lr_delay_steps_(0), lr_delay_mult_(1.0), max_steps_(1000000)
{
    this->max_sh_degree_ = sh_degree;

    // Device
    if (torch::cuda::is_available())
        this->device_type_ = torch::kCUDA;
    else
        this->device_type_ = torch::kCPU;

    GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)
}

GaussianModel::GaussianModel(const GaussianModelParams &model_params)
    : active_sh_degree_(0), spatial_lr_scale_(0.0),
      lr_delay_steps_(0), lr_delay_mult_(1.0), max_steps_(1000000)
{
    this->max_sh_degree_ = model_params.sh_degree_;

    // Device
    if (model_params.data_device_ == "cuda")
        this->device_type_ = torch::kCUDA;
    else
        this->device_type_ = torch::kCPU;

    GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)
}

torch::Tensor GaussianModel::getScalingActivation()
{
    return torch::exp(this->scaling_);
}

torch::Tensor GaussianModel::getRotationActivation()
{
    return torch::nn::functional::normalize(this->rotation_);
}

torch::Tensor GaussianModel::getXYZ()
{
    return this->xyz_;
}

torch::Tensor GaussianModel::getFeatures()
{
    return torch::cat({this->features_dc_.clone(), this->features_rest_.clone()}, /*dim=*/1);
}

torch::Tensor GaussianModel::getOpacityActivation()
{
    return torch::sigmoid(this->opacity_);
}

torch::Tensor GaussianModel::getCovarianceActivation(int scaling_modifier)
{
    // build_rotation
    auto r = this->rotation_;
    auto R = general_utils::build_rotation(r);

    // build_scaling_rotation(scaling_modifier * scaling(Activation), rotation(_))
    auto s = scaling_modifier * this->getScalingActivation();
    auto L = torch::zeros({s.size(0), 3, 3}, torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    L.select(1, 0).select(1, 0).copy_(s.index({torch::indexing::Slice(), 0}));
    L.select(1, 1).select(1, 1).copy_(s.index({torch::indexing::Slice(), 1}));
    L.select(1, 2).select(1, 2).copy_(s.index({torch::indexing::Slice(), 2}));
    L = R.matmul(L); // L = R @ L

    // build_covariance_from_scaling_rotation
    auto actual_covariance = L.matmul(L.transpose(1, 2));
    // strip_symmetric
    // strip_lowerdiag
    auto symm_uncertainty = torch::zeros({actual_covariance.size(0), 6}, torch::TensorOptions().dtype(torch::kFloat).device(device_type_));

    symm_uncertainty.select(1, 0).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 0}));
    symm_uncertainty.select(1, 1).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 1}));
    symm_uncertainty.select(1, 2).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 2}));
    symm_uncertainty.select(1, 3).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 1}));
    symm_uncertainty.select(1, 4).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 2}));
    symm_uncertainty.select(1, 5).copy_(actual_covariance.index({torch::indexing::Slice(), 2, 2}));

    return symm_uncertainty;
}

void GaussianModel::oneUpShDegree()
{
    if (this->active_sh_degree_ < this->max_sh_degree_)
        this->active_sh_degree_ += 1;
}

void GaussianModel::setShDegree(const int sh)
{
    this->active_sh_degree_ = (sh > this->max_sh_degree_ ? this->max_sh_degree_ : sh);
}

void GaussianModel::createFromPcd(
    std::map<point3D_id_t, Point3D> pcd,
    const float spatial_lr_scale)
{
    this->spatial_lr_scale_ = spatial_lr_scale;
    int num_points = static_cast<int>(pcd.size());
    torch::Tensor fused_point_cloud = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    torch::Tensor color = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    auto pcd_it = pcd.begin();
    for (int point_idx = 0; point_idx < num_points; ++point_idx) {
        auto& point = (*pcd_it).second;
        fused_point_cloud.index({point_idx, 0}) = point.xyz_(0);
        fused_point_cloud.index({point_idx, 1}) = point.xyz_(1);
        fused_point_cloud.index({point_idx, 2}) = point.xyz_(2);
        color.index({point_idx, 0}) = point.color_(0);
        color.index({point_idx, 1}) = point.color_(1);
        color.index({point_idx, 2}) = point.color_(2);
        ++pcd_it;
    }

    torch::Tensor fused_color = sh_utils::RGB2SH(color);
    auto temp = this->max_sh_degree_ + 1;
    torch::Tensor features = torch::zeros(
        {fused_color.size(0), 3, temp * temp},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(0, 3),
         0}) = fused_color;
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(3, features.size(1)),
         torch::indexing::Slice(1, features.size(2))}) = 0.0f;

    // std::cout << "[Gaussian Model]Number of points at initialization : " << fused_point_cloud.size(0) << std::endl;

    torch::Tensor point_cloud_copy = fused_point_cloud.clone();
    torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
    torch::Tensor scales = torch::log(torch::sqrt(dist2));
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
    torch::Tensor rots = torch::zeros({fused_point_cloud.size(0), 4}, torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;

    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {fused_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    this->exist_since_iter_ = torch::zeros(
        {fused_point_cloud.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    this->xyz_ = fused_point_cloud.requires_grad_();
    this->features_dc_ = features.index({torch::indexing::Slice(),
                                         torch::indexing::Slice(),
                                         torch::indexing::Slice(0, 1)})
                             .transpose(1, 2)
                             .contiguous()
                             .requires_grad_();
    this->features_rest_ = features.index({torch::indexing::Slice(),
                                           torch::indexing::Slice(),
                                           torch::indexing::Slice(1, features.size(2))})
                               .transpose(1, 2)
                               .contiguous()
                               .requires_grad_();
    this->scaling_ = scales.requires_grad_();
    this->rotation_ = rots.requires_grad_();
    this->opacity_ = opacities.requires_grad_();

    GAUSSIAN_MODEL_TENSORS_TO_VEC

    this->max_radii2D_ = torch::zeros({this->getXYZ().size(0)}, torch::TensorOptions().device(device_type_));
}

void GaussianModel::increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration)
{
// auto time1 = std::chrono::steady_clock::now();
    assert(points.size() == colors.size());
    assert(points.size() % 3 == 0);
    auto num_new_points = static_cast<int>(points.size() / 3);
    if (num_new_points == 0)
        return;

    torch::Tensor new_point_cloud = torch::from_blob(
        points.data(), {num_new_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        // torch::zeros({num_new_points, 3}, xyz_.options());
    torch::Tensor new_colors = torch::from_blob(
        colors.data(), {num_new_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        // torch::zeros({num_new_points, 3}, xyz_.options());

    if (sparse_points_xyz_.size(0) == 0) {
        sparse_points_xyz_ = new_point_cloud;
        sparse_points_color_ = new_colors;
    }
    else {
        sparse_points_xyz_ = torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
        sparse_points_color_ = torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);
    }

    torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
    auto temp = this->max_sh_degree_ + 1;
    torch::Tensor features = torch::zeros(
        {new_fused_colors.size(0), 3, temp * temp},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(0, 3),
         0}) = new_fused_colors;
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(3, features.size(1)),
         torch::indexing::Slice(1, features.size(2))}) = 0.0f;

    // std::cout << "[Gaussian Model]Number of points increase : "
    //           << num_new_points << std::endl;

    torch::Tensor dist2 = torch::clamp_min(
        distCUDA2(new_point_cloud.clone()), 0.0000001);
    torch::Tensor scales = torch::log(torch::sqrt(dist2));
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
    torch::Tensor rots = torch::zeros(
        {new_point_cloud.size(0), 4},
         torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {new_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    torch::Tensor new_exist_since_iter = torch::full(
        {new_point_cloud.size(0)},
        iteration,
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    auto new_xyz = new_point_cloud;
    auto new_features_dc = features.index({torch::indexing::Slice(),
                                                    torch::indexing::Slice(),
                                                    torch::indexing::Slice(0, 1)})
                                        .transpose(1, 2)
                                        .contiguous();
    auto new_features_rest = features.index({torch::indexing::Slice(),
                                                      torch::indexing::Slice(),
                                                      torch::indexing::Slice(1, features.size(2))})
                                          .transpose(1, 2)
                                          .contiguous();
    auto new_opacities = opacities;
    auto new_scaling = scales;
    auto new_rotation = rots;

// auto time2 = std::chrono::steady_clock::now();
// auto time = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
// std::cout << "increasePcd(umap) preparation time: " << time << " ms" <<std::endl;

    densificationPostfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_exist_since_iter
    );

    c10::cuda::CUDACachingAllocator::emptyCache();
// auto time3 = std::chrono::steady_clock::now();
// time = std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
// std::cout << "increasePcd(umap) postfix time: " << time << " ms" <<std::endl;
}

void GaussianModel::increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration)
{
// auto time1 = std::chrono::steady_clock::now();
    auto num_new_points = new_point_cloud.size(0);
    if (num_new_points == 0)
        return;

    if (sparse_points_xyz_.size(0) == 0) {
        sparse_points_xyz_ = new_point_cloud;
        sparse_points_color_ = new_colors;
    }
    else {
        sparse_points_xyz_ = torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
        sparse_points_color_ = torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);
    }

    torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
    auto temp = this->max_sh_degree_ + 1;
    torch::Tensor features = torch::zeros(
        {new_fused_colors.size(0), 3, temp * temp},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(0, 3),
         0}) = new_fused_colors;
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(3, features.size(1)),
         torch::indexing::Slice(1, features.size(2))}) = 0.0f;

    // std::cout << "[Gaussian Model]Number of points increase : "
    //           << num_new_points << std::endl;

    torch::Tensor dist2 = torch::clamp_min(
        distCUDA2(new_point_cloud.clone()), 0.0000001);
    torch::Tensor scales = torch::log(torch::sqrt(dist2));
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
    torch::Tensor rots = torch::zeros(
        {new_point_cloud.size(0), 4},
         torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {new_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    torch::Tensor new_exist_since_iter = torch::full(
        {new_point_cloud.size(0)},
        iteration,
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    auto new_xyz = new_point_cloud;
    auto new_features_dc = features.index({torch::indexing::Slice(),
                                                    torch::indexing::Slice(),
                                                    torch::indexing::Slice(0, 1)})
                                        .transpose(1, 2)
                                        .contiguous();
    auto new_features_rest = features.index({torch::indexing::Slice(),
                                                      torch::indexing::Slice(),
                                                      torch::indexing::Slice(1, features.size(2))})
                                          .transpose(1, 2)
                                          .contiguous();
    auto new_opacities = opacities;
    auto new_scaling = scales;
    auto new_rotation = rots;

// auto time2 = std::chrono::steady_clock::now();
// auto time = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
// std::cout << "increasePcd(tensor) preparation time: " << time << " ms" <<std::endl;

    densificationPostfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_exist_since_iter
    );

    c10::cuda::CUDACachingAllocator::emptyCache();

// auto time3 = std::chrono::steady_clock::now();
// time = std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
// std::cout << "increasePcd(tensor) postfix time: " << time << " ms" <<std::endl;
}

void GaussianModel::applyScaledTransformation(
    const float s,
    const Sophus::SE3f T)
{
    torch::NoGradGuard no_grad;
    // pt <- (s * Ryw * pt + tyw)
    this->xyz_ *= s;
    torch::Tensor T_tensor =
        tensor_utils::EigenMatrix2TorchTensor(T.matrix(), device_type_).transpose(0, 1);
    transformPoints(this->xyz_, T_tensor);

// torch::Tensor scales;
// torch::Tensor point_cloud_copy = this->xyz_.clone();
// torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
// scales = torch::log(torch::sqrt(dist2));
// auto scales_ndimension = scales.ndimension();
// scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
    this->scaling_ *= s;
    scaledTransformationPostfix(this->xyz_, this->scaling_);
}

void GaussianModel::scaledTransformationPostfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_scaling)
{
    // param_groups[0] = xyz_
    torch::Tensor optimizable_xyz = this->replaceTensorToOptimizer(new_xyz, 0);
    // param_groups[4] = scaling_
    torch::Tensor optimizable_scaling = this->replaceTensorToOptimizer(new_scaling, 4);

    this->xyz_ = optimizable_xyz;
    this->scaling_ = optimizable_scaling;

    this->Tensor_vec_xyz_ = {this->xyz_};
    this->Tensor_vec_scaling_ = {this->scaling_};
}

void GaussianModel::scaledTransformVisiblePointsOfKeyframe(
    torch::Tensor& point_not_transformed_flags,
    torch::Tensor& diff_pose,
    torch::Tensor& kf_world_view_transform,
    torch::Tensor& kf_full_proj_transform,
    const int kf_creation_iter,
    const int stable_num_iter_existence,
    int& num_transformed,
    const float scale)
{
    torch::NoGradGuard no_grad;

    torch::Tensor points = this->getXYZ();
    torch::Tensor rots = this->getRotationActivation();
    // torch::Tensor scales = this->scaling_;// * scale;

    torch::Tensor point_unstable_flags = torch::where(
        torch::abs(this->exist_since_iter_ - kf_creation_iter) < stable_num_iter_existence,
        true,
        false);

    scaleAndTransformThenMarkVisiblePoints(
        points,
        rots,
        point_not_transformed_flags,
        point_unstable_flags,
        diff_pose,
        kf_world_view_transform,
        kf_full_proj_transform,
        num_transformed,
        scale
    );

// torch::Tensor point_cloud_copy = points.clone();
// torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
// torch::Tensor scales = torch::log(torch::sqrt(dist2));
// auto scales_ndimension = scales.ndimension();
// scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});

    // Postfix
    // ==================================
    // param_groups[0] = xyz_
    // param_groups[1] = feature_dc_
    // param_groups[2] = feature_rest_
    // param_groups[3] = opacity_
    // param_groups[4] = scaling_
    // param_groups[5] = rotation_
    // ==================================
    torch::Tensor optimizable_xyz = this->replaceTensorToOptimizer(points, 0);
    // torch::Tensor optimizable_scaling = this->replaceTensorToOptimizer(scales, 4);
    torch::Tensor optimizable_rots = this->replaceTensorToOptimizer(rots, 5);

    this->xyz_ = optimizable_xyz;
    // this->scaling_ = optimizable_scaling;
    this->rotation_ = optimizable_rots;

    this->Tensor_vec_xyz_ = {this->xyz_};
    // this->Tensor_vec_scaling_ = {this->scaling_};
    this->Tensor_vec_rotation_ = {this->rotation_};
}

void GaussianModel::trainingSetup(const GaussianOptimizationParams& training_args)
{
    setPercentDense(training_args.percent_dense_);
    this->xyz_gradient_accum_ = torch::zeros({this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
    this->denom_ = torch::zeros({this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));

    torch::optim::AdamOptions adam_options;
    adam_options.set_lr(0.0);
    adam_options.eps() = 1e-15;

    this->optimizer_.reset(new torch::optim::Adam(Tensor_vec_xyz_, adam_options));
    optimizer_->param_groups()[0].options().set_lr(training_args.position_lr_init_ * this->spatial_lr_scale_);

    optimizer_->add_param_group(Tensor_vec_feature_dc_);
    optimizer_->param_groups()[1].options().set_lr(training_args.feature_lr_);

    optimizer_->add_param_group(Tensor_vec_feature_rest_);
    optimizer_->param_groups()[2].options().set_lr(training_args.feature_lr_ / 20.0);

    optimizer_->add_param_group(Tensor_vec_opacity_);
    optimizer_->param_groups()[3].options().set_lr(training_args.opacity_lr_);

    optimizer_->add_param_group(Tensor_vec_scaling_);
    optimizer_->param_groups()[4].options().set_lr(training_args.scaling_lr_);

    optimizer_->add_param_group(Tensor_vec_rotation_);
    optimizer_->param_groups()[5].options().set_lr(training_args.rotation_lr_);

    // get_expon_lr_func
    lr_init_ = training_args.position_lr_init_ * this->spatial_lr_scale_;
    lr_final_ = training_args.position_lr_final_ * this->spatial_lr_scale_;
    lr_delay_mult_ = training_args.position_lr_delay_mult_;
    max_steps_ = training_args.position_lr_max_steps_;
}

float GaussianModel::updateLearningRate(int step)
{
    // def update_learning_rate(self, iteration):
    //     ''' Learning rate scheduling per step '''
    //     for param_group in self.optimizer.param_groups:
    //         if param_group["name"] == "xyz":
    //             lr = self.xyz_scheduler_args(iteration)
    //             param_group['lr'] = lr
    //             return lr
    float lr = this->exponLrFunc(step);
    optimizer_->param_groups()[0].options().set_lr(lr); // Tensor_vec_xyz_
    return lr;
}

// ==================================
// param_groups[0] = xyz_
// param_groups[1] = feature_dc_
// param_groups[2] = feature_rest_
// param_groups[3] = opacity_
// param_groups[4] = scaling_
// param_groups[5] = rotation_
// ==================================
void GaussianModel::setPositionLearningRate(float position_lr)
{
    optimizer_->param_groups()[0].options().set_lr(position_lr * this->spatial_lr_scale_);
}
void GaussianModel::setFeatureLearningRate(float feature_lr)
{
    optimizer_->param_groups()[1].options().set_lr(feature_lr);
    optimizer_->param_groups()[2].options().set_lr(feature_lr / 20.0);
}
void GaussianModel::setOpacityLearningRate(float opacity_lr)
{
    optimizer_->param_groups()[3].options().set_lr(opacity_lr);
}
void GaussianModel::setScalingLearningRate(float scaling_lr)
{
    optimizer_->param_groups()[4].options().set_lr(scaling_lr);
}
void GaussianModel::setRotationLearningRate(float rot_lr)
{
    optimizer_->param_groups()[5].options().set_lr(rot_lr);
}

void GaussianModel::resetOpacity()
{
    torch::Tensor opacities_new = general_utils::inverse_sigmoid(
        torch::min(
            this->getOpacityActivation(),
            torch::ones_like(this->getOpacityActivation() * 0.01)));
    torch::Tensor optimizable_tensors = this->replaceTensorToOptimizer(opacities_new, 3); // "opacity"
    this->opacity_ = optimizable_tensors;
    this->Tensor_vec_opacity_ = {this->opacity_};
}

torch::Tensor GaussianModel::replaceTensorToOptimizer(torch::Tensor& tensor, int tensor_idx)
{
    auto& param = this->optimizer_->param_groups()[tensor_idx].params()[0];
    auto& state = optimizer_->state();
    auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
    auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(stored_state.step());
    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));
    // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); // needed only when options.amsgrad(true), which is false by default

    state.erase(key);
    param = tensor.requires_grad_();
    key = c10::guts::to_string(param.unsafeGetTensorImpl());
    state[key] = std::move(new_state);

    auto optimizable_tensors = param;
    return optimizable_tensors;
}

void GaussianModel::prunePoints(torch::Tensor& mask)
{
    auto valid_points_mask = ~mask;

    // _prune_optimizer
    std::vector<torch::Tensor> optimizable_tensors(6);
    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();
    for (int group_idx = 0; group_idx < 6; ++group_idx) {
        auto& param = param_groups[group_idx].params()[0];
        auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
        if (state.find(key) != state.end()) {
            auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(stored_state.step());
            new_state->exp_avg(stored_state.exp_avg().index({valid_points_mask}).clone());
            new_state->exp_avg_sq(stored_state.exp_avg_sq().index({valid_points_mask}).clone());
            // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); // needed only when options.amsgrad(true), which is false by default

            state.erase(key);
            param = param.index({valid_points_mask}).requires_grad_();
            key = c10::guts::to_string(param.unsafeGetTensorImpl());
            state[key] = std::move(new_state);
            optimizable_tensors[group_idx] = param;
        }
        else {
            param = param.index({valid_points_mask}).requires_grad_();
            optimizable_tensors[group_idx] = param;
        }
    }

    // ==================================
    // param_groups[0] = xyz_
    // param_groups[1] = feature_dc_
    // param_groups[2] = feature_rest_
    // param_groups[3] = opacity_
    // param_groups[4] = scaling_
    // param_groups[5] = rotation_
    // ==================================
    this->xyz_ = optimizable_tensors[0];
    this->features_dc_ = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_ = optimizable_tensors[3];
    this->scaling_ = optimizable_tensors[4];
    this->rotation_ = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC

    this->exist_since_iter_ = this->exist_since_iter_.index({valid_points_mask});

    this->xyz_gradient_accum_ = this->xyz_gradient_accum_.index({valid_points_mask});

    this->denom_ = this->denom_.index({valid_points_mask});
    this->max_radii2D_ = this->max_radii2D_.index({valid_points_mask});
}

void GaussianModel::densificationPostfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation,
    torch::Tensor& new_exist_since_iter)
{
    // cat_tensors_to_optimizer
    std::vector<torch::Tensor> optimizable_tensors(6);
    std::vector<torch::Tensor> tensors_dict = {
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation
    };
    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();
    for (int group_idx = 0; group_idx < 6; ++group_idx) {
        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);
        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];
        auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
        if (state.find(key) != state.end()) {
            auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(stored_state.step());
            new_state->exp_avg(torch::cat({stored_state.exp_avg().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));
            new_state->exp_avg_sq(torch::cat({stored_state.exp_avg_sq().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));
            // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone());  // needed only when options.amsgrad(true), which is false by default

            state.erase(key);
            param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
            key = c10::guts::to_string(param.unsafeGetTensorImpl());
            state[key] = std::move(new_state);

            optimizable_tensors[group_idx] = param;
        }
        else {
            param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
            optimizable_tensors[group_idx] = param;
        }
    }

    // ==================================
    // param_groups[0] = xyz_
    // param_groups[1] = feature_dc_
    // param_groups[2] = feature_rest_
    // param_groups[3] = opacity_
    // param_groups[4] = scaling_
    // param_groups[5] = rotation_
    // ==================================
    this->xyz_ = optimizable_tensors[0];
    this->features_dc_ = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_ = optimizable_tensors[3];
    this->scaling_ = optimizable_tensors[4];
    this->rotation_ = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC

    this->exist_since_iter_ = torch::cat({this->exist_since_iter_, new_exist_since_iter}, /*dim=*/0);

    this->xyz_gradient_accum_ = torch::zeros({this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
    this->denom_ = torch::zeros({this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
    this->max_radii2D_ = torch::zeros({this->getXYZ().size(0)}, torch::TensorOptions().device(device_type_));
}

void GaussianModel::densifyAndSplit(
    torch::Tensor& grads,
    float grad_threshold,
    float scene_extent,
    int N)
{
    int n_init_points = this->getXYZ().size(0);
    // Extract points that satisfy the gradient condition
    auto padded_grad = torch::zeros({n_init_points}, torch::TensorOptions().device(device_type_));
    padded_grad.slice(/*dim=*/0L, /*start=*/0, /*end=*/grads.size(0)).copy_(grads.squeeze());
    auto selected_pts_mask = torch::where(padded_grad >= grad_threshold, true, false);
    selected_pts_mask = torch::logical_and(
        selected_pts_mask,
        std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) > percentDense() * scene_extent
    );

    auto stds = this->getScalingActivation().index({selected_pts_mask}).repeat({N, 1});
    auto means = torch::zeros({stds.size(0), 3}, torch::TensorOptions().device(device_type_));
    auto samples = at::normal(means, stds);
    auto r_masked = this->rotation_.index({selected_pts_mask});
    auto rots = general_utils::build_rotation(r_masked).repeat({N, 1, 1});
    auto new_xyz = torch::bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + this->getXYZ().index({selected_pts_mask}).repeat({N, 1});
    auto new_scaling = torch::log(this->getScalingActivation().index({selected_pts_mask}).repeat({N, 1}) / (0.8 * N)); // scaling_inverse_activation
    auto new_rotation = this->rotation_.index({selected_pts_mask}).repeat({N, 1});
    auto new_features_dc = this->features_dc_.index({selected_pts_mask}).repeat({N, 1, 1});
    auto new_features_rest = this->features_rest_.index({selected_pts_mask}).repeat({N, 1, 1});
    auto new_opacity = this->opacity_.index({selected_pts_mask}).repeat({N, 1});

    auto new_exist_since_iter = this->exist_since_iter_.index({selected_pts_mask}).repeat({N});

    this->densificationPostfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        new_exist_since_iter
    );

    auto prune_filter = torch::cat({
        selected_pts_mask,
        torch::zeros({(N * selected_pts_mask.sum()).item<int>()}, torch::TensorOptions().device(device_type_).dtype(torch::kBool))
    });
    this->prunePoints(prune_filter);
}

void GaussianModel::densifyAndClone(
    torch::Tensor& grads,
    float grad_threshold,
    float scene_extent)
{
    // Extract points that satisfy the gradient condition
    auto selected_pts_mask = torch::where(torch::frobenius_norm(grads, /*dim=*/-1) >= grad_threshold, true, false);
    selected_pts_mask = torch::logical_and(
        selected_pts_mask,
        std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) <= percentDense() * scene_extent
    );

    auto new_xyz = this->xyz_.index({selected_pts_mask});
    auto new_features_dc = this->features_dc_.index({selected_pts_mask});
    auto new_features_rest = this->features_rest_.index({selected_pts_mask});
    auto new_opacities = this->opacity_.index({selected_pts_mask});
    auto new_scaling = this->scaling_.index({selected_pts_mask});
    auto new_rotation = this->rotation_.index({selected_pts_mask});

    auto new_exist_since_iter = this->exist_since_iter_.index({selected_pts_mask});

    this->densificationPostfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_exist_since_iter
    );
}

void GaussianModel::densifyAndPrune(
    float max_grad,
    float min_opacity,
    float extent,
    int max_screen_size)
{
    auto grads = this->xyz_gradient_accum_ / this->denom_;
    grads.index_put_({grads.isnan()}, 0.0f);
    this->densifyAndClone(grads, max_grad, extent);
    this->densifyAndSplit(grads, max_grad, extent);

    auto prune_mask = (this->getOpacityActivation() < min_opacity).squeeze();
    if (max_screen_size) {
        auto big_points_vs = this->max_radii2D_ > max_screen_size;
        auto big_points_ws = std::get<0>(this->getScalingActivation().max(/*dim=*/1)) > 0.1f * extent;
        prune_mask = torch::logical_or(torch::logical_or(prune_mask, big_points_vs), big_points_ws);
    }
    this->prunePoints(prune_mask);

    c10::cuda::CUDACachingAllocator::emptyCache(); // torch.cuda.empty_cache()
}

void GaussianModel::addDensificationStats(
    torch::Tensor& viewspace_point_tensor,
    torch::Tensor& update_filter)
{
    this->xyz_gradient_accum_.index_put_(
        {update_filter},
        torch::frobenius_norm(viewspace_point_tensor.grad().index({update_filter, torch::indexing::Slice(0, 2)}),
                              /*dim=*/-1,
                              /*keepdim=*/true),
        /*accumulate=*/true);

    this->denom_.index_put_(
        {update_filter},
        this->denom_.index({update_filter}) + 1);
}

// void GaussianModel::increasePointsIterationsOfExistence(const int i)
// {
//     this->exist_since_iter_ += i;
// }

void GaussianModel::loadPly(std::filesystem::path ply_path)
{
    std::ifstream instream_binary(ply_path, std::ios::binary);
    if (!instream_binary.is_open() || instream_binary.fail())
        throw std::runtime_error("Fail to open ply file at " + ply_path.string());
    instream_binary.seekg(0, std::ios::beg);

    tinyply::PlyFile ply_file;
    ply_file.parse_header(instream_binary);

    std::cout << "\t[ply_header] Type: " << (ply_file.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto & c : ply_file.get_comments())
        std::cout << "\t[ply_header] Comment: " << c << std::endl;
    for (const auto & c : ply_file.get_info())
        std::cout << "\t[ply_header] Info: " << c << std::endl;

    for (const auto &e : ply_file.get_elements()) {
        std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
        for (const auto &p : e.properties) {
            std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
            if (p.isList)
                std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
            std::cout << std::endl;
        }
    }

    std::shared_ptr<tinyply::PlyData> xyz, f_dc, f_rest, opacity, scales, rot;

    try { xyz = ply_file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { f_dc = ply_file.request_properties_from_element("vertex", { "f_dc_0", "f_dc_1", "f_dc_2" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    int n_f_rest = ((max_sh_degree_ + 1) * (max_sh_degree_ + 1) - 1) * 3;
    if (n_f_rest >= 0) {
        std::vector<std::string> f_rest_element_names(n_f_rest);
        for (int i = 0; i < n_f_rest; ++i)
            f_rest_element_names[i] = "f_rest_" + std::to_string(i);
        try {f_rest = ply_file.request_properties_from_element("vertex", f_rest_element_names); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
    }

    try { opacity = ply_file.request_properties_from_element("vertex", { "opacity" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { scales = ply_file.request_properties_from_element("vertex", { "scale_0", "scale_1", "scale_2" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { rot = ply_file.request_properties_from_element("vertex", { "rot_0", "rot_1", "rot_2", "rot_3" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    ply_file.read(instream_binary);

    if (xyz)     std::cout << "\tRead " << xyz->count     << " total xyz "     << std::endl;
    if (f_dc)    std::cout << "\tRead " << f_dc->count    << " total f_dc "    << std::endl;
    if (f_rest)  std::cout << "\tRead " << f_rest->count  << " total f_rest "  << std::endl;
    if (opacity) std::cout << "\tRead " << opacity->count << " total opacity " << std::endl;
    if (scales)  std::cout << "\tRead " << scales->count  << " total scales "  << std::endl;
    if (rot)     std::cout << "\tRead " << rot->count     << " total rot "     << std::endl;

    // Data to std::vector
    const int num_points = xyz->count;

    const std::size_t n_xyz_bytes = xyz->buffer.size_bytes();
    std::vector<float> xyz_vector(xyz->count * 3);
    std::memcpy(xyz_vector.data(), xyz->buffer.get(), n_xyz_bytes);

    const std::size_t n_f_dc_bytes = f_dc->buffer.size_bytes();
    std::vector<float> f_dc_vector(f_dc->count * 3);
    std::memcpy(f_dc_vector.data(), f_dc->buffer.get(), n_f_dc_bytes);

    const std::size_t n_f_rest_bytes = f_rest->buffer.size_bytes();
    std::vector<float> f_rest_vector(f_rest->count * n_f_rest);
    std::memcpy(f_rest_vector.data(), f_rest->buffer.get(), n_f_rest_bytes);

    const std::size_t n_opacity_bytes = opacity->buffer.size_bytes();
    std::vector<float> opacity_vector(opacity->count * 1);
    std::memcpy(opacity_vector.data(), opacity->buffer.get(), n_opacity_bytes);

    const std::size_t n_scales_bytes = scales->buffer.size_bytes();
    std::vector<float> scales_vector(scales->count * 3);
    std::memcpy(scales_vector.data(), scales->buffer.get(), n_scales_bytes);

    const std::size_t n_rot_bytes = rot->buffer.size_bytes();
    std::vector<float> rot_vector(rot->count * 4);
    std::memcpy(rot_vector.data(), rot->buffer.get(), n_rot_bytes);

    // std::vector to torch::Tensor
    this->xyz_ = torch::from_blob(
        xyz_vector.data(), {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    this->features_dc_ = torch::from_blob(
        f_dc_vector.data(), {num_points, 3, 1},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_).transpose(1, 2).contiguous();

    this->features_rest_ = torch::from_blob(
        f_rest_vector.data(), {num_points, 3, n_f_rest / 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_).transpose(1, 2).contiguous();

    this->opacity_ = torch::from_blob(
        opacity_vector.data(), {num_points, 1},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    this->scaling_ = torch::from_blob(
        scales_vector.data(), {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    this->rotation_ = torch::from_blob(
        rot_vector.data(), {num_points, 4},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    GAUSSIAN_MODEL_TENSORS_TO_VEC

    this->active_sh_degree_ = this->max_sh_degree_;
}

void GaussianModel::savePly(std::filesystem::path result_path)
{
    // Prepare data to write
    torch::Tensor xyz = this->xyz_.detach().cpu();
    torch::Tensor normals = torch::zeros_like(xyz);
    torch::Tensor f_dc = this->features_dc_.detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor f_rest = this->features_rest_.detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor opacities = this->opacity_.detach().cpu();
    torch::Tensor scale = this->scaling_.detach().cpu();
    torch::Tensor rotation = this->rotation_.detach().cpu();

    std::filebuf fb_binary;
    fb_binary.open(result_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + result_path.string());

    tinyply::PlyFile result_file;

    // xyz
    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, xyz.size(0),
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // normals
    result_file.add_properties_to_element(
        "vertex", {"nx", "ny", "nz"},
        tinyply::Type::FLOAT32, normals.size(0),
        reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // f_dc
    std::size_t n_f_dc = this->features_dc_.size(1) * this->features_dc_.size(2);
    std::vector<std::string> property_names_f_dc(n_f_dc);
    for (int i = 0; i < n_f_dc; ++i)
        property_names_f_dc[i] = "f_dc_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_dc,
        tinyply::Type::FLOAT32, this->features_dc_.size(0),
        reinterpret_cast<uint8_t*>(f_dc.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // f_rest
    std::size_t n_f_rest = this->features_rest_.size(1) * this->features_rest_.size(2);
    std::vector<std::string> property_names_f_rest(n_f_rest);
    for (int i = 0; i < n_f_rest; ++i)
        property_names_f_rest[i] = "f_rest_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_rest,
        tinyply::Type::FLOAT32, this->features_rest_.size(0),
        reinterpret_cast<uint8_t*>(f_rest.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // opacities
    result_file.add_properties_to_element(
        "vertex", {"opacity"},
        tinyply::Type::FLOAT32, opacities.size(0),
        reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // scale
    std::size_t n_scale = scale.size(1);
    std::vector<std::string> property_names_scale(n_scale);
    for (int i = 0; i < n_scale; ++i)
        property_names_scale[i] = "scale_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_scale,
        tinyply::Type::FLOAT32, scale.size(0),
        reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // rotation
    std::size_t n_rotation = rotation.size(1);
    std::vector<std::string> property_names_rotation(n_rotation);
    for (int i = 0; i < n_rotation; ++i)
        property_names_rotation[i] = "rot_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_rotation,
        tinyply::Type::FLOAT32, rotation.size(0),
        reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // Write the file
    result_file.write(outstream_binary, true);

    fb_binary.close();
}

void GaussianModel::saveSparsePointsPly(std::filesystem::path result_path)
{
    // Prepare data to write
    torch::Tensor xyz = this->sparse_points_xyz_.detach().cpu();
    torch::Tensor normals = torch::zeros_like(xyz);
    torch::Tensor color = (this->sparse_points_color_ * 255.0f).toType(torch::kUInt8).detach().cpu();

    std::filebuf fb_binary;
    fb_binary.open(result_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + result_path.string());

    tinyply::PlyFile result_file;

    // xyz
    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, xyz.size(0),
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // normals
    result_file.add_properties_to_element(
        "vertex", {"nx", "ny", "nz"},
        tinyply::Type::FLOAT32, normals.size(0),
        reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // color
    result_file.add_properties_to_element(
        "vertex", {"red", "green", "blue"},
        tinyply::Type::UINT8, color.size(0),
        reinterpret_cast<uint8_t*>(color.data_ptr<uint8_t>()),
        tinyply::Type::INVALID, 0);

    // Write the file
    result_file.write(outstream_binary, true);

    fb_binary.close();
}

float GaussianModel::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return percent_dense_;
}

void GaussianModel::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    percent_dense_ = percent_dense;
}

/**
 * @brief get_expon_lr_func
 * @details Modified from Plenoxels
 *  Continuous learning rate decay function. Adapted from JaxNeRF
 *  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
 *  is log-linearly interpolated elsewhere (equivalent to exponential decay).
 *  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
 *  function of lr_delay_mult, such that the initial learning rate is
 *  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
 *  to the normal learning rate when steps>lr_delay_steps.
 *  :param conf: config subtree 'lr' or similar
 *  :param max_steps: int, the number of steps during optimization.
 *  :return HoF which takes step as input
 * @param iteration 
 * @return float 
 */
float GaussianModel::exponLrFunc(int step)
{
    if (step < 0 || (lr_init_ == 0.0f && lr_final_ == 0.0f))
        return 0.0f;

    float delay_rate;
    if (lr_delay_steps_ > 0)
        delay_rate = lr_delay_mult_ + (1.0f - lr_delay_mult_) * std::sin(M_PI_2f32 * std::clamp(static_cast<float>(step) / lr_delay_steps_, 0.0f, 1.0f));
    else
        delay_rate = 1.0f;
    float t = std::clamp(static_cast<float>(step) / max_steps_, 0.0f, 1.0f);
    float log_lerp = std::exp(std::log(lr_init_) * (1 - t) + std::log(lr_final_) * t);
    return delay_rate * log_lerp;
}