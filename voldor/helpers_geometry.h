
#pragma once

#include <Eigen/Eigen>

// OK
// For M cameras and N points
// P:      3x4*M
// p2d:    2*MxN
// return: 4xN
template <typename A, typename B>
Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> triangulate(Eigen::MatrixBase<A> const& P, Eigen::MatrixBase<B> const& p2d)
{
    Eigen::Index const M = P.cols() / 4;
    Eigen::Index const N = p2d.cols();

    Eigen::Matrix<typename A::Scalar, 2, 3> L{ {0, -1, 0}, {1, 0, 0} };

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> ls_matrix(2 * M, 4);
    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> p3d_h(4, N);

    for (int n = 0; n < N; ++n)
    {
    for (int m = 0; m < M; ++m)
    {
    L(0, 2) =  p2d((2 * m) + 1, n);
    L(1, 2) = -p2d((2 * m) + 0, n);

    ls_matrix(Eigen::seqN(2 * m, 2), Eigen::all) = L * P(Eigen::all, Eigen::seqN(4 * m, 4));
    }

    p3d_h.col(n) = ls_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3); // Previously BDC SVD, Full = Thin
    }

    return p3d_h;
}

// OK
// For N points
template <typename _scalar> struct result_R_t_from_E
{
    Eigen::Matrix<_scalar, Eigen::Dynamic, Eigen::Dynamic> P;     // 3x4
    Eigen::Matrix<_scalar, Eigen::Dynamic, Eigen::Dynamic> p3d_h; // 4xN
};

// OK
// For N points
// E:     3x3
// p2d_1: 2xN
// p2d_2: 2xN
template <typename A, typename B, typename C>
result_R_t_from_E<typename A::Scalar> R_t_from_E(Eigen::MatrixBase<A> const& E, Eigen::MatrixBase<B> const& p2d_1, Eigen::MatrixBase<C> const& p2d_2)
{
    Eigen::Index const N = p2d_1.cols();

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> p2d(4, N);

    p2d(Eigen::seqN(0, 2), Eigen::all) = p2d_1;
    p2d(Eigen::seqN(2, 2), Eigen::all) = p2d_2;

    Eigen::Matrix<typename A::Scalar, 3, 3> W{ {0, -1, 0}, {1, 0, 0}, {0, 0, 1} };

    Eigen::JacobiSVD<typename A::PlainObject> E_svd = E.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV); // Full = Thin

    Eigen::Matrix<typename A::Scalar, 3, 3> U  = E_svd.matrixU();
    Eigen::Matrix<typename A::Scalar, 3, 3> Vt = E_svd.matrixV().transpose();

    Eigen::Matrix<typename A::Scalar, 3, 3> R1 = U * W             * Vt;
    Eigen::Matrix<typename A::Scalar, 3, 3> R2 = U * W.transpose() * Vt;

    if (R1.determinant() < 0) { R1 = -R1; } // Self assign OK
    if (R2.determinant() < 0) { R2 = -R2; } // Self assign OK

    Eigen::Matrix<typename A::Scalar, 3, 1> pt = U.col(2);
    Eigen::Matrix<typename A::Scalar, 3, 1> nt = -pt;

    Eigen::Matrix<typename A::Scalar, 3, 4> P1 = Eigen::Matrix<typename A::Scalar, 3, 4>::Identity();
    Eigen::Matrix<typename A::Scalar, 3, 4> P2;
    Eigen::Matrix<typename A::Scalar, 3, 8> PX;

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> XYZW;

    int64_t max_count = -1;
    result_R_t_from_E<typename A::Scalar> result;
    
    for (int i = 0; i < 4; ++i)
    {
    switch (i)
    {
    case 0: P2 << R1, pt; break;
    case 1: P2 << R1, nt; break;
    case 2: P2 << R2, pt; break;
    case 3: P2 << R2, nt; break;
    }

    PX(Eigen::all, Eigen::seqN(0, 4)) = P1;
    PX(Eigen::all, Eigen::seqN(4, 4)) = P2;

    XYZW = triangulate(PX, p2d).colwise().hnormalized().colwise().homogeneous();

    int64_t count = ((P1 * XYZW).row(2).array() > 0).count() + ((P2 * XYZW).row(2).array() > 0).count();
    if (count < max_count) { continue; }
    max_count = count;

    result.P     = P2;
    result.p3d_h = XYZW;
    }

    return result;
}

// OK
// E = [e11 e12 e13; e21, e22, e23; e31, e32, e33]
// e = [e11 e21 e31 e12 e22 e32 e13 e23 e33]
// For N points
// p2dh_1: 3xN
// p2dh_2: 3xN
// return: Nx9
template <typename A, typename B>
Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_E_constraints(Eigen::MatrixBase<A> const& p2dh_1, Eigen::MatrixBase<B> const& p2dh_2)
{
    Eigen::Index const N = p2dh_1.cols();

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> Q(N, 9);

    Q.col(0) = p2dh_1.row(0).cwiseProduct(p2dh_2.row(0)).transpose();
    Q.col(1) = p2dh_1.row(0).cwiseProduct(p2dh_2.row(1)).transpose();
    Q.col(2) = p2dh_1.row(0).cwiseProduct(p2dh_2.row(2)).transpose();
    Q.col(3) = p2dh_1.row(1).cwiseProduct(p2dh_2.row(0)).transpose();
    Q.col(4) = p2dh_1.row(1).cwiseProduct(p2dh_2.row(1)).transpose();
    Q.col(5) = p2dh_1.row(1).cwiseProduct(p2dh_2.row(2)).transpose();
    Q.col(6) = p2dh_1.row(2).cwiseProduct(p2dh_2.row(0)).transpose();
    Q.col(7) = p2dh_1.row(2).cwiseProduct(p2dh_2.row(1)).transpose();
    Q.col(8) = p2dh_1.row(2).cwiseProduct(p2dh_2.row(2)).transpose();

    return Q;
}


















template <typename _scalar, int _rows, int _cols>
static Eigen::Matrix<_scalar, _rows, _cols> matrix_R_cayley(_scalar kx, _scalar ky, _scalar kz)
{
    Eigen::Matrix<_scalar, _rows, _cols> R(3, 3);

    R << 1 + kx * kx - ky * ky - kz * kz, 2 * kx * ky - 2 * kz, 2 * kx * kz + 2 * ky,
         2 * kx * ky + 2 * kz, 1 - kx * kx + ky * ky - kz * kz, 2 * ky * kz - 2 * kx,
         2 * kx * kz - 2 * ky, 2 * ky * kz + 2 * kx, 1 - kx * kx - ky * ky + kz * kz;
    R /= 1 + kx * kx + ky * ky + kz * kz;

    return R;
}

template <typename _scalar, int _rows_1, int _cols_1, int _rows_2, int _cols_2>
Eigen::Matrix<_scalar, _rows_1, _cols_1> cross_matrix(Eigen::Matrix<_scalar, _rows_2, _cols_2> const& v)
{
    Eigen::Matrix<_scalar, _rows_1, _cols_1> M(3, 3);

    M(0, 0) = 0;
    M(1, 0) =  v(2);
    M(2, 0) = -v(1);
    M(0, 1) = -v(2);
    M(1, 1) = 0;
    M(2, 1) =  v(0);
    M(0, 2) =  v(1);
    M(1, 2) = -v(0);
    M(2, 2) = 0;

    return M;
}

template <typename _scalar, int _rows, int _cols>
Eigen::Matrix<_scalar, _rows, _cols> normalize_essential(Eigen::Matrix<_scalar, _rows, _cols> const& E)
{
    Eigen::JacobiSVD<Eigen::Matrix<_scalar, _rows, _cols>> E_svd = E.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    return E_svd.matrixU() * Eigen::Matrix<_scalar, _rows, _cols>{ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0 } } * E_svd.matrixV().transpose();
}
