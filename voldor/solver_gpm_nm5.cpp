
#include <Eigen/Eigen>
#include "polynomial.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"

bool solver_gpm_nm5(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12)
{
    Eigen::Matrix<float, 3, 5> P1 = matrix_from_buffer<float, 3, 5>(p3d_1);
    Eigen::Matrix<float, 3, 5> P2 = matrix_from_buffer<float, 3, 5>(p2h_2);

    Eigen::Matrix<float, 2, 5> q1 = P1.colwise().hnormalized();
    Eigen::Matrix<float, 2, 5> q2 = P2.colwise().hnormalized();

    Eigen::Matrix<float, 3, 5> Q1 = q1.colwise().homogeneous();
    Eigen::Matrix<float, 3, 5> Q2 = q2.colwise().homogeneous();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q = matrix_E_constraints(Q1, Q2);

    Q.col(4) -= Q.col(0);
    Q.col(8) -= Q.col(0);

    Eigen::Matrix<float, 8, 3> k = Q(Eigen::indexing::all, Eigen::seqN(1, 8)).fullPivLu().kernel();
    Eigen::Matrix<float, 9, 3> e;

    e << (-(k(3, Eigen::indexing::all) + k(7, Eigen::indexing::all))), k;

    Eigen::Matrix<x38::polynomial<float, 2>, 3, 3> E = x38::matrix_to_polynomial_grevlex<float, 2, 3, 3>(e); // OK

    x38::polynomial<float, 2> E_determinant = E.determinant();

    Eigen::Matrix<x38::polynomial<float, 2>, 3, 3> EEt = E * E.transpose();
    Eigen::Matrix<x38::polynomial<float, 2>, 3, 3> E_singular_values = (EEt * E) - ((0.5 * EEt.trace()) * E);

    Eigen::Matrix<float, 10, 10> S;

    S << x38::matrix_from_polynomial_grevlex<float, 9, 10>(E_singular_values),
         x38::matrix_from_polynomial_grevlex<float, 1, 10>(E_determinant);

    Eigen::Matrix<float, 10, 1> solution = S.bdcSvd<Eigen::ComputeThinV>().matrixV().col(9);

    Eigen::Matrix<float, 3, 3> fake_E = (e(Eigen::indexing::all, 0) + ((solution(1) / solution(0)) * e(Eigen::indexing::all, 1)) + ((solution(2) / solution(0)) * e(Eigen::indexing::all, 2))).reshaped(3, 3);

    result_R_t_from_E result = R_t_from_E(fake_E, P1, q2);

    Eigen::Matrix<float, 3, 3> R = result.P(Eigen::indexing::all, Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 1> t = result.P.col(3);

    Eigen::Matrix<float, 3, 1> r = vector_r_rodrigues(R);

    matrix_to_buffer(r, r_12);
    matrix_to_buffer(t, t_12);

    return is_valid_pose(r, t);
}
