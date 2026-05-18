
#pragma once

#include <PoseLib/misc/sturm.h>

template <int degree>
int find_real_roots(float const* factors, float* roots)
{
    std::unique_ptr<double[]> f = std::make_unique<double[]>(degree + 1);
    std::unique_ptr<double[]> r = std::make_unique<double[]>(degree);
    for (int i = 0; i < (degree + 1); ++i) { f[i] = factors[i]; }
    int nroots = poselib::sturm::bisect_sturm<degree>(f.get(), r.get());
    for (int i = 0; i < nroots; ++i) { roots[i] = static_cast<float>(r[i]); }
    return nroots;
}
