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

#include <Eigen/Geometry>

namespace graphics_utils
{

inline int roundToIntegerMultipleOf16(int integer)
{
    int remainder = integer % 16;

    if (remainder == 0) {
        return integer;
    }
    else if (remainder < 8) {
        return integer - remainder;
    }
    else {
        return integer - remainder + 16;
    }

    return integer;
}

inline float fov2focal(float fov, int pixels)
{
    return pixels / (2.0f * std::tan(fov / 2.0f));
}

inline float focal2fov(float focal, int pixels)
{
    return 2.0f * std::atan(pixels / (2.0f * focal));
}

}
