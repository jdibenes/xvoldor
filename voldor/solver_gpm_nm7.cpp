
#include <Eigen/Eigen>
#include "helpers_eigen.h"
#include "helpers_geometry.h"

bool solver_gpm_nm7(float const* p1, float const* p2, float* r01, float* t01)
{
    Eigen::Matrix<float, 3, 7> P1 = matrix_from_buffer<float, 3, 7>(p1);
    Eigen::Matrix<float, 3, 7> P2 = matrix_from_buffer<float, 3, 7>(p2);

    Eigen::Matrix<float, 2, 7> q1 = P1.colwise().hnormalized();
    Eigen::Matrix<float, 2, 7> q2 = P2.colwise().hnormalized();

    Eigen::Matrix<float, 3, 7> Q1 = q1.colwise().homogeneous();
    Eigen::Matrix<float, 3, 7> Q2 = q2.colwise().homogeneous();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q = matrix_E_constraints(Q1, Q2);

    Q.col(4) = Q.col(4) - Q.col(0);
    Q.col(8) = Q.col(8) - Q.col(0);

    Eigen::Matrix<float, 8, 1> k = Q(Eigen::indexing::all, Eigen::seqN(1, 8)).fullPivLu().kernel();
    Eigen::Matrix<float, 9, 1> e;

    e << (-(k(3) + k(7))), k;

    Eigen::Matrix<float, 3, 3> fake_E = e.reshaped(3, 3);
    
    result_R_t_from_E result = R_t_from_E(fake_E, P1, q2);

    Eigen::Matrix<float, 3, 3> R = result.P(Eigen::indexing::all, Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 1> t = result.P.col(3);

    Eigen::Matrix<float, 3, 1> r = vector_r_rodrigues(R);

    matrix_to_buffer(r, r01);
    matrix_to_buffer(t, t01);

    return is_valid_pose(r, t);
}
