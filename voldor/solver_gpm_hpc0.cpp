
#include <Eigen/Eigen>
#include "helpers.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"

bool solver_gpm_hpc0(float const* p3d_1, float const* p3d_2, float* r_12, float* t_12)
{
    Eigen::Matrix<float, 3, 2> P1 = matrix_from_buffer<float, 3, 2>(p3d_1);
    Eigen::Matrix<float, 3, 2> P2 = matrix_from_buffer<float, 3, 2>(p3d_2);

    Eigen::Matrix<float, 3, 1> PA1 = P1.col(0);
    Eigen::Matrix<float, 3, 1> PB1 = P1.col(1);
    Eigen::Matrix<float, 3, 1> PA2 = P2.col(0);
    Eigen::Matrix<float, 3, 1> PB2 = P2.col(1);

    Eigen::Matrix<float, 3, 1> V1 = (PA1 - PB1).normalized();
    Eigen::Matrix<float, 3, 1> V2 = (PA2 - PB2).normalized();

    float thetaF = acos_small(clamp(V1.dot(V2), -1.0f, 1.0f));

    Eigen::Matrix<float, 3, 1> kF = V1.cross(V2).normalized();

    Eigen::Matrix<float, 3, 1> DA = PA1 - PA2;
    Eigen::Matrix<float, 3, 1> DB = PB1 - PB2;

    float LA = DA.norm();
    float LB = DB.norm();

    Eigen::Matrix<float, 3, 1> UA = DA / LA;
    Eigen::Matrix<float, 3, 1> UB = DB / LB;

    float tF_2 = thetaF / 2.0f;

    float cF_2 = std::cos(tF_2);
    float sF_2 = std::sin(tF_2);

    Eigen::Matrix<float, 3, 1> kG = kF.cross(V1);

    float cc;
    float cs;

    if (LA > LB) { cc = -sF_2 * UA.dot(kF); cs = (cF_2 * UA.dot(V1)) + (sF_2 * UA.dot(kG)); }
    else         { cc = -sF_2 * UB.dot(kF); cs = (cF_2 * UB.dot(V1)) + (sF_2 * UB.dot(kG)); }

    Eigen::Matrix<float, 3, 1> kR = ((sF_2 * cs * kF) + (cF_2 * cc * V1) + (sF_2 * cc * kG)).normalized();

    float thetaR = 2.0f * acos_small(clamp((cs * cF_2) / std::sqrt((cc * cc) + (cs * cs)), -1.0f, 1.0f));

    Eigen::Matrix<float, 3, 1> r = kR * thetaR;
    Eigen::Matrix<float, 3, 1> t = ((PA2 + PB2) - (matrix_R_rodrigues(r) * (PA1 + PB1))) / 2.0f;

    matrix_to_buffer(r, r_12);
    matrix_to_buffer(t, t_12);

    return is_valid_pose(r, t);
}
