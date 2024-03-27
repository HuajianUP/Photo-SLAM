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

namespace general_utils
{

inline torch::Tensor inverse_sigmoid(const torch::Tensor &x)
{
    return torch::log(x / (1 - x));
}

inline torch::Tensor build_rotation(torch::Tensor &r)
{
    auto r0 = r.index({torch::indexing::Slice(), 0});
    auto r1 = r.index({torch::indexing::Slice(), 1});
    auto r2 = r.index({torch::indexing::Slice(), 2});
    auto r3 = r.index({torch::indexing::Slice(), 3});
    auto norm = torch::sqrt(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3);

    auto q = r / norm.unsqueeze(/*dim=*/1);
    r = q.index({torch::indexing::Slice(), 0});
    auto x = q.index({torch::indexing::Slice(), 1});
    auto y = q.index({torch::indexing::Slice(), 2});
    auto z = q.index({torch::indexing::Slice(), 3});

    auto R = torch::zeros({q.size(0), 3, 3}, torch::TensorOptions().device(torch::kCUDA));
    R.select(1, 0).select(1, 0).copy_(1 - 2 * (y * y + z * z));
    R.select(1, 0).select(1, 1).copy_(2 * (x * y - r * z));
    R.select(1, 0).select(1, 2).copy_(2 * (x * z + r * y));
    R.select(1, 1).select(1, 0).copy_(2 * (x * y + r * z));
    R.select(1, 1).select(1, 1).copy_(1 - 2 * (x * x + z * z));
    R.select(1, 1).select(1, 2).copy_(2 * (y * z - r * x));
    R.select(1, 2).select(1, 0).copy_(2 * (x * z - r * y));
    R.select(1, 2).select(1, 1).copy_(2 * (y * z + r * x));
    R.select(1, 2).select(1, 2).copy_(1 - 2 * (x * x + y * y));
    return R;
}

}
