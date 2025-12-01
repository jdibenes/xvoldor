
#include <Eigen/Eigen>
#include "solver_gpm_hpc0.h"
#include "helpers.h"
#include "helpers_eigen.h"

bool solver_gpm_hpc1(float const* pa1, float const* pb1, float const* pa2, float const* pb2, float* r01, float* t01)
{
    Eigen::Matrix<float, 3, 1> PA1 = matrix_from_buffer<float, 3, 1>(pa1);
    Eigen::Matrix<float, 3, 1> PB1 = matrix_from_buffer<float, 3, 1>(pb1);
    Eigen::Matrix<float, 3, 1> PA2 = matrix_from_buffer<float, 3, 1>(pa2);
    Eigen::Matrix<float, 3, 1> PB2 = matrix_from_buffer<float, 3, 1>(pb2);

    Eigen::Matrix<float, 3, 1> PB2_n = PB2 / PB2(2, 0);
    Eigen::Matrix<float, 3, 1> PX1_d = PB1 - PA1;
    
    float a = PB2_n.dot(PB2_n);
    float b = -2.0f * PB2_n.dot(PA2);
    float c = PA2.dot(PA2) - PX1_d.dot(PX1_d);

    float d = (b * b) - (4.0f * a * c);
    float s = (d > 0.0f) ? std::sqrt(d) : 0.0f;
    float q = 2.0f * a;

    float z1 = ((-b + s) / q);
    float z2 = ((-b - s) / q);

    Eigen::Matrix<float, 3, 1> PB2_w = PB2_n * ((std::abs(PB2(2, 0) - z1) <= std::abs(PB2(2, 0) - z2)) ? z1 : z2);

    return solver_gpm_hpc1(PA1.data(), PB1.data(), PA2.data(), PB2_w.data(), r01, t01);
}
