
#include <Eigen/Eigen>
#include "helpers_eigen.h"
#include "helpers_geometry.h"

bool solver_tft_linear(float const* p3d_1, float const* p2d_2, float const* p2d_3, int N, float* r_12, float* t_12, float* r_23, float* t_23, float threshold)
{
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> p1 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p3d_1, 3, N);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> p2 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2d_2, 2, N);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> p3 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2d_3, 2, N);

    result_normalize_points rnp_1 = normalize_points(p1.colwise().hnormalized());
    result_normalize_points rnp_2 = normalize_points(p2);
    result_normalize_points rnp_3 = normalize_points(p3);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A = matrix_TFT_constraints(rnp_1.p, rnp_2.p, rnp_3.p); // OK
    result_linear_TFT<float> ltr = linear_TFT(A, threshold); // OK
    ltr.TFT = transform_TFT(ltr.TFT, rnp_1.H, rnp_2.H, rnp_3.H, true); // OK
    result_R_t_from_TFT<float> rpt = R_t_from_TFT(ltr.TFT, p1, p2, p3); // OK

    Eigen::Matrix<float, 3, 3> R12 = rpt.v2.P(Eigen::indexing::all, Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 3> R13 = rpt.v3.P(Eigen::indexing::all, Eigen::seqN(0, 3));

    Eigen::Matrix<float, 3, 1> t12 = rpt.v2.P.col(3);
    Eigen::Matrix<float, 3, 1> t13 = rpt.v3.P.col(3);

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
