// #  Copyright 2021 The PlenOctree Authors.
// #  Redistribution and use in source and binary forms, with or without
// #  modification, are permitted provided that the following conditions are met:
// #
// #  1. Redistributions of source code must retain the above copyright notice,
// #  this list of conditions and the following disclaimer.
// #
// #  2. Redistributions in binary form must reproduce the above copyright notice,
// #  this list of conditions and the following disclaimer in the documentation
// #  and/or other materials provided with the distribution.
// #
// #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// #  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// #  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// #  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// #  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// #  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// #  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// #  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// #  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// #  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// #  POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <array>

#include <torch/torch.h>

namespace sh_utils
{

const float C0 = 0.28209479177387814f;
const float C1 = 0.4886025119029199f;
const std::array<float, 5> C2 = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
const std::array<float, 7> C3 = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};
const std::array<float, 9> C4 = {
    2.5033429417967046f,
    -1.7701307697799304f,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f
};


inline torch::Tensor eval_sh(int deg, torch::Tensor& sh, torch::Tensor& dirs)
{

    // """
    // Evaluate spherical harmonics at unit directions
    // using hardcoded SH polynomials.
    // Works with torch/np/jnp.
    // ... Can be 0 or more batch dimensions.
    // Args:
    //     deg: int SH deg. Currently, 0-3 supported
    //     sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
    //     dirs: jnp.ndarray unit directions [..., 3]
    // Returns:
    //     [..., C]
    // """
    assert(deg <= 4 && deg >= 0);
    int coeff = (deg + 1) * (deg + 1);
    assert(sh.size(-1) >= coeff);

    auto result = C0 * sh.index({torch::indexing::Ellipsis, 0});
    if (deg > 0)
    {
        auto x = dirs.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 1)});
        auto y = dirs.index({torch::indexing::Ellipsis, torch::indexing::Slice(1, 2)});
        auto z = dirs.index({torch::indexing::Ellipsis, torch::indexing::Slice(2, 3)});
        result = (result -
                C1 * y * sh.index({torch::indexing::Ellipsis, 1}) +
                C1 * z * sh.index({torch::indexing::Ellipsis, 2}) -
                C1 * x * sh.index({torch::indexing::Ellipsis, 3}));

        if (deg > 1)
        {
            auto xx = x * x;
            auto yy = y * y;
            auto zz = z * z;
            auto xy = x * y;
            auto yz = y * z;
            auto xz = x * z;
            result = (result +
                    C2[0] * xy * sh.index({torch::indexing::Ellipsis, 4}) +
                    C2[1] * yz * sh.index({torch::indexing::Ellipsis, 5}) +
                    C2[2] * (2.0f * zz - xx - yy) * sh.index({torch::indexing::Ellipsis, 6}) +
                    C2[3] * xz * sh.index({torch::indexing::Ellipsis, 7}) +
                    C2[4] * (xx - yy) * sh.index({torch::indexing::Ellipsis, 8}));

            if (deg > 2)
            {
                result = (result +
                        C3[0] * y * (3 * xx - yy) * sh.index({torch::indexing::Ellipsis, 9}) +
                        C3[1] * xy * z * sh.index({torch::indexing::Ellipsis, 10}) +
                        C3[2] * y * (4 * zz - xx - yy)* sh.index({torch::indexing::Ellipsis, 11}) +
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh.index({torch::indexing::Ellipsis, 12}) +
                        C3[4] * x * (4 * zz - xx - yy) * sh.index({torch::indexing::Ellipsis, 13}) +
                        C3[5] * z * (xx - yy) * sh.index({torch::indexing::Ellipsis, 14}) +
                        C3[6] * x * (xx - 3 * yy) * sh.index({torch::indexing::Ellipsis, 15}));

                if (deg > 3)
                {
                    result = (result + C4[0] * xy * (xx - yy) * sh.index({torch::indexing::Ellipsis, 16}) +
                            C4[1] * yz * (3 * xx - yy) * sh.index({torch::indexing::Ellipsis, 17}) +
                            C4[2] * xy * (7 * zz - 1) * sh.index({torch::indexing::Ellipsis, 18}) +
                            C4[3] * yz * (7 * zz - 3) * sh.index({torch::indexing::Ellipsis, 19}) +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh.index({torch::indexing::Ellipsis, 20}) +
                            C4[5] * xz * (7 * zz - 3) * sh.index({torch::indexing::Ellipsis, 21}) +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh.index({torch::indexing::Ellipsis, 22}) +
                            C4[7] * xz * (xx - 3 * yy) * sh.index({torch::indexing::Ellipsis, 23}) +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh.index({torch::indexing::Ellipsis, 24}));
                }
            }
        }
    }
    return result;
}

inline torch::Tensor RGB2SH(torch::Tensor& rgb)
{
    return (rgb - 0.5f) / C0;
}

inline float SH2RGB(float sh)
{
    return sh * C0 + 0.5f;
}

}
