
#include <limits>
#include <Eigen/Eigen>
#include "polynomial.h"
#include "algebra.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"

static bool solver_rpe_easy(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12, bool planar)
{
    int const hidden_variable_index = 2; // 0: x, 1: y, 2: z
    int const points = planar ? 4 : 5;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> P1 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p3d_1, 3, points);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> P2 = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2h_2, 3, points);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> q1 = P1.colwise().hnormalized();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> q2 = P2.colwise().hnormalized();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q1 = q1.colwise().homogeneous();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q2 = q2.colwise().homogeneous();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q = matrix_E_constraints(Q1, Q2);

    Eigen::Matrix<float, 9, 4> e;

    if (planar)
    {
    Q.col(4) -= Q.col(0);
    Q.col(8) -= Q.col(0);

    Eigen::Matrix<float, 8, 4> k = Q(Eigen::indexing::all, Eigen::seqN(1, 8)).fullPivLu().kernel();

    e << (-(k(3, Eigen::indexing::all) + k(7, Eigen::indexing::all))), k;
    }
    else
    {
    e = Q.fullPivLu().kernel();
    }

    Eigen::Matrix<x38::polynomial<float, 3>, 3, 3> E = x38::matrix_to_polynomial_grevlex<float, 3, 3, 3>(e);

    Eigen::Matrix<x38::polynomial<x38::polynomial<float, 1>, 2>, 3, 3> E_hidden = x38::hide_in(E, hidden_variable_index);

    x38::polynomial<x38::polynomial<float, 1>, 2> E_determinant = E_hidden.determinant();

    Eigen::Matrix<x38::polynomial<x38::polynomial<float, 1>, 2>, 3, 3> EEt = E_hidden * E_hidden.transpose();
    Eigen::Matrix<x38::polynomial<x38::polynomial<float, 1>, 2>, 3, 3> E_singular_values = (EEt * E_hidden) - ((0.5 * EEt.trace()) * E_hidden);

    Eigen::Matrix<x38::polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> S(10, 10);

    S << x38::matrix_from_polynomial_grevlex<x38::polynomial<float, 1>, 9, 10>(E_singular_values).rowwise().reverse(),
         x38::matrix_from_polynomial_grevlex<x38::polynomial<float, 1>, 1, 10>(E_determinant).rowwise().reverse();

    for (int i = 0; i < 4; ++i) { x38::row_echelon_step(S, i, i, { 0 }, false); }

    Eigen::Matrix<x38::polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> S6 = S(Eigen::seqN(4, 6), Eigen::seqN(4, 6));

    x38::row_echelon_step(S6, 0, 0, { 1 }, true);
    x38::row_echelon_step(S6, 1, 0, { 0 }, true);

    x38::row_echelon_step(S6, 2, 1, { 1 }, true);
    x38::row_echelon_step(S6, 3, 1, { 0 }, true);

    x38::row_echelon_step(S6, 4, 2, { 1 }, true);
    x38::row_echelon_step(S6, 5, 2, { 0 }, true);

    x38::polynomial<float, 1> hidden_variable = x38::monomial<float, 1>{ 1, { 1 } };

    S6.row(1) = (S6.row(1) * hidden_variable) - S6.row(0);
    S6.row(3) = (S6.row(3) * hidden_variable) - S6.row(2);
    S6.row(5) = (S6.row(5) * hidden_variable) - S6.row(4);

    S6.row(1).swap(S6.row(2));
    S6.row(2).swap(S6.row(4));

    Eigen::Matrix<x38::polynomial<float, 1>, 3, 3> S3 = S6(Eigen::seqN(3, 3), Eigen::seqN(3, 3));

    x38::polynomial<float, 1> hidden_univariate = S3.determinant();

    Eigen::Matrix<float, 1, 11> polynomial = x38::matrix_from_polynomial_grevlex<float, 1, 11>(hidden_univariate);

    polynomial.normalize();

    float roots[10];
    int nroots = find_real_roots(polynomial.data(), 10, roots);
    if (nroots <= 0) { return false; }

    result_R_t_from_E<float> result;

    Eigen::Matrix<float, 3, 3> R;
    Eigen::Matrix<float, 3, 1> t;

    float rerror = std::numeric_limits<float>::infinity();
    bool  set    = false;

    for (int i = 0; i < nroots; ++i)
    {
    float z = roots[i];

    Eigen::Matrix<float, 10, 1> monomial_eigenvector = x38::slice(x38::substitute(S, { true }, x38::monomial_values<float, 1>{ z }), {}).bdcSvd<Eigen::ComputeFullV>().matrixV().col(9);

    float x = monomial_eigenvector(8) / monomial_eigenvector(9);
    float y = monomial_eigenvector(7) / monomial_eigenvector(9);

    Eigen::Matrix<float, 3, 3> E_estimated = x38::slice(x38::substitute(E, { true, true, true }, x38::merge(x38::array_type<float, 2>{ x, y }, hidden_variable_index, z)), {});

    result = R_t_from_E(E_estimated, P1, q2);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> p2 = result.P * result.p3h;

    if ((p2.row(2).array() <= 0).count() > 0) { continue; }

    float re = (q2 - p2.colwise().hnormalized()).colwise().norm().sum();

    if (!std::isfinite(re) || (re >= rerror)) { continue; }

    R = result.P(Eigen::indexing::all, Eigen::seqN(0, 3));
    t = result.P.col(3);

    rerror = re;
    set    = true;
    }

    if (!set) { return false; }

    Eigen::Matrix<float, 3, 1> r = vector_r_rodrigues(R);

    matrix_to_buffer(r, r_12);
    matrix_to_buffer(t, t_12);

    return is_valid_pose(r, t);
}

bool solver_gpm_m4(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12)
{
    return solver_rpe_easy(p3d_1, p2h_2, r_12, t_12, true);
}

bool solver_rpe_m5(float const* p3d_1, float const* p2h_2, float* r_12, float* t_12)
{
    return solver_rpe_easy(p3d_1, p2h_2, r_12, t_12, false);
}
