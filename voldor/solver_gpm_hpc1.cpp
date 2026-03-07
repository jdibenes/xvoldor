
#include <Eigen/Eigen>
#include "solver_gpm.h"
#include "algebra.h"
#include "helpers.h"
#include "helpers_eigen.h"

bool solver_gpm_hpc1(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12)
{
    Eigen::Matrix<float, 3, 2> P1 = matrix_from_buffer<float, 3, 2>(p3d_1);
    Eigen::Matrix<float, 3, 2> P2 = matrix_from_buffer<float, 3, 2>(p3d_2);

    Eigen::Matrix<float, 3, 1> PA1 = P1.col(0);
    Eigen::Matrix<float, 3, 1> PB1 = P1.col(1);
    Eigen::Matrix<float, 3, 1> PA2 = P2.col(0);
    Eigen::Matrix<float, 3, 1> PB2 = P2.col(1);

    Eigen::Matrix<float, 3, 1> PB2_n = PB2 / PB2(2, 0);
    Eigen::Matrix<float, 3, 1> PX1_d = PB1 - PA1;

    Eigen::Matrix<float, 3, 1> polynomial{ PA2.dot(PA2) - PX1_d.dot(PX1_d), -2.0f * PB2_n.dot(PA2), PB2_n.dot(PB2_n) };

    polynomial.normalize();

    float roots[2];
    int nroots = find_real_roots(polynomial.data(), 2, roots);
    if (nroots <= 0) { return false; }

    float z1 = roots[0];
    float z2 = roots[1];

    P2.col(1) = PB2_n * ((std::abs(PB2(2, 0) - z1) <= std::abs(PB2(2, 0) - z2)) ? z1 : z2);

    return solver_gpm_hpc0(P1.data(), P2.data(), r_12, t_12);
}
