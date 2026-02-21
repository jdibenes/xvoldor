
#include <Eigen/Eigen>
#include "solver_gpm_hpc0.h"
#include "algebra.h"
#include "helpers.h"
#include "helpers_eigen.h"

bool solver_gpm_hpc1(float const* p1, float const* p2, float* r01, float* t01, int refine_iterations)
{
    Eigen::Matrix<float, 3, 2> P1 = matrix_from_buffer<float, 3, 2>(p1);
    Eigen::Matrix<float, 3, 2> P2 = matrix_from_buffer<float, 3, 2>(p2);

    Eigen::Matrix<float, 3, 1> PA1 = P1.col(0);
    Eigen::Matrix<float, 3, 1> PB1 = P1.col(1);
    Eigen::Matrix<float, 3, 1> PA2 = P2.col(0);
    Eigen::Matrix<float, 3, 1> PB2 = P2.col(1);

    Eigen::Matrix<float, 3, 1> PB2_n = PB2 / PB2(2, 0);
    Eigen::Matrix<float, 3, 1> PX1_d = PB1 - PA1;

    float cba[3] = { PA2.dot(PA2) - PX1_d.dot(PX1_d), -2.0f * PB2_n.dot(PA2), PB2_n.dot(PB2_n) };
    float roots[2];

    solve_quadratic(cba, roots, refine_iterations);

    float z1 = roots[0];
    float z2 = roots[1];

    Eigen::Matrix<float, 3, 1> PB2_w = PB2_n * ((std::abs(PB2(2, 0) - z1) <= std::abs(PB2(2, 0) - z2)) ? z1 : z2);

    return solver_gpm_hpc0(PA1.data(), PB1.data(), PA2.data(), PB2_w.data(), r01, t01);
}
