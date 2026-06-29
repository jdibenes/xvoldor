
#include <Eigen/Eigen>
#include "helpers_eigen.h"

bool detect_planar_3d(float const* p3d_1, float const* p3d_2, float* out, float threshold_zero, float threshold_sin)
{
    Eigen::Matrix<float, 3, 3> P1 = matrix_from_buffer<float, 3, 3>(p3d_1);
    Eigen::Matrix<float, 3, 3> P2 = matrix_from_buffer<float, 3, 3>(p3d_2);

    Eigen::Matrix<float, 3, 3> D = P2 - P1;

    Eigen::Matrix<float, 3, 1> a = D.col(0);
    Eigen::Matrix<float, 3, 1> b = D.col(1);
    Eigen::Matrix<float, 3, 1> c = D.col(2);

    float n_a = a.norm();
    float n_b = b.norm();
    float n_c = c.norm();

    if (n_a <= threshold_zero) { return false; }
    if (n_b <= threshold_zero) { return false; }
    if (n_c <= threshold_zero) { return false; }

    a /= n_a;
    b /= n_b;
    c /= n_c;

    Eigen::Matrix<float, 3, 1> k = a.cross(b);

    float sin_ab = k.norm();
    if (sin_ab <= threshold_sin) { return false; }

    k /= sin_ab;
    float cos_kc = std::abs(k.dot(c));

    *out = cos_kc;
    return std::isfinite(cos_kc);
}
