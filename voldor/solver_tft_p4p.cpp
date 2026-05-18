
#include <Eigen/Eigen>
#include <opencv2/calib3d.hpp>
#include <lambdatwist/lambdatwist_p4p.h>
#include "helpers_eigen.h"
#include "helpers_geometry.h"

bool solver_tft_p4p(float const* p3d_1, float const* p2d_2, float const* p2d_3, float* r_12, float* t_12, float* r_23, float* t_23)
{
    cv::Vec3f const* p1 = reinterpret_cast<cv::Vec3f const*>(p3d_1);
    cv::Vec2f const* q2 = reinterpret_cast<cv::Vec2f const*>(p2d_2);
    cv::Vec2f const* q3 = reinterpret_cast<cv::Vec2f const*>(p2d_3);

    bool ok;

    float R12_s[3][3];
    cv::Vec3f t12_s;

    ok = lambdatwist_p4p<double, float, 5>(q2[0].val, q2[1].val, q2[2].val, q2[3].val, p1[0].val, p1[1].val, p1[2].val, p1[3].val, 1, 1, 0, 0, R12_s, t12_s.val);
    if (!ok) { return false; }

    float R13_s[3][3];
    cv::Vec3f t13_s;

    ok = lambdatwist_p4p<double, float, 5>(q3[0].val, q3[1].val, q3[2].val, q3[3].val, p1[0].val, p1[1].val, p1[2].val, p1[3].val, 1, 1, 0, 0, R13_s, t13_s.val);
    if (!ok) { return false; }

    Eigen::Matrix<float, 3, 3> R12 = matrix_from_buffer<float, 3, 3>(&R12_s[0][0]).transpose();
    Eigen::Matrix<float, 3, 1> t12 = matrix_from_buffer<float, 3, 1>(t12_s.val);
    Eigen::Matrix<float, 3, 3> R13 = matrix_from_buffer<float, 3, 3>(&R13_s[0][0]).transpose();
    Eigen::Matrix<float, 3, 1> t13 = matrix_from_buffer<float, 3, 1>(t13_s.val);

    Eigen::Matrix<float, 3, 3> R23 = R13 * R12.transpose();
    Eigen::Matrix<float, 3, 1> t23 = t13 - R23 * t12;

    Eigen::Matrix<float, 3, 1> r12 = vector_r_rodrigues(R12);
    Eigen::Matrix<float, 3, 1> r23 = vector_r_rodrigues(R23);

    matrix_to_buffer(r12, r_12);
    matrix_to_buffer(t12, t_12);
    matrix_to_buffer(r23, r_23);
    matrix_to_buffer(t23, t_23);

    return is_valid_pose(r12, t12) && is_valid_pose(r23, t23);
}
