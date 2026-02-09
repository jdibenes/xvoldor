
#include <Eigen/Eigen>
#include "polynomial.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"

bool solver_gpm_nm5(float const* p1, float const* p2, float* r01, float* t01)
{
    Eigen::Matrix<float, 3, 5> P1 = matrix_from_buffer<float, 3, 5>(p1);
    Eigen::Matrix<float, 3, 5> P2 = matrix_from_buffer<float, 3, 5>(p2);

    Eigen::Matrix<float, 2, 5> q1 = P1.colwise().hnormalized();
    Eigen::Matrix<float, 2, 5> q2 = P2.colwise().hnormalized();

    Eigen::Matrix<float, 3, 5> Q1 = q1.colwise().homogeneous();
    Eigen::Matrix<float, 3, 5> Q2 = q2.colwise().homogeneous();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q = matrix_E_constraints(Q1, Q2);

    Q.col(4) = Q.col(4) - Q.col(0);
    Q.col(8) = Q.col(8) - Q.col(0);

    Eigen::Matrix<float, 8, 3> k = Q(Eigen::all, Eigen::seqN(1, 8)).fullPivLu().kernel();
    Eigen::Matrix<float, 9, 3> e;

    e << (-(k(3, Eigen::all) + k(7, Eigen::all))), k;

    Eigen::Matrix<polynomial<float, 2>, 3, 3> E = matrix_to_polynomial_grevlex<float, 2, 3, 3>(e); // OK

    polynomial<float, 2> E_determinant = E.determinant();

    Eigen::Matrix<polynomial<float, 2>, 3, 3> EEt = E * E.transpose();
    Eigen::Matrix<polynomial<float, 2>, 3, 3> E_singular_values = (EEt * E) - ((0.5 * EEt.trace()) * E);

    Eigen::Matrix<float, 10, 10> S;

    S << matrix_from_polynomial_grevlex<float, 9, 10>(E_singular_values),
         matrix_from_polynomial_grevlex<float, 1, 10>(E_determinant);

    Eigen::Matrix<float, 10, 1> solution = S.bdcSvd(Eigen::ComputeThinV).matrixV().col(9);

    Eigen::Matrix<float, 3, 3> fake_E = (e(Eigen::all, 0) + ((solution(1) / solution(0)) * e(Eigen::all, 1)) + ((solution(2) / solution(0)) * e(Eigen::all, 2))).reshaped(3, 3);

    result_R_t_from_E result = R_t_from_E(fake_E, q1, q2);

    Eigen::Matrix<float, 3, 3> R = result.P(Eigen::all, Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 1> v = result.P.col(3);

    Eigen::AngleAxis<float> aa(R);

    Eigen::Matrix<float, 3, 1> r = aa.axis() * aa.angle();
    Eigen::Matrix<float, 3, 1> t = (P2.col(0) - R * P1.col(0)).norm() * v;

    matrix_to_buffer(r, r01);
    matrix_to_buffer(t, t01);

    float r_sum = r01[0] + r01[1] + r01[2];
    float t_sum = t01[0] + t01[1] + t01[2];

    float x_sum = r_sum + t_sum;

    return std::isfinite(x_sum);
}
