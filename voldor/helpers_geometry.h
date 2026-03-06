
#pragma once

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <unsupported/Eigen/KroneckerProduct>

// OK
// p3d_1: 3xN
// p2d_2: 2xN
// P2:    3x4
template <typename A, typename B, typename C>
typename A::Scalar compute_scale(Eigen::MatrixBase<A> const& p3d_1, Eigen::MatrixBase<B> const& p2d_2, Eigen::MatrixBase<C> const& P2)
{
    Eigen::Index const N = p3d_1.cols();

    Eigen::Matrix<typename A::Scalar, 3, Eigen::Dynamic> X3 = P2(Eigen::indexing::all, Eigen::seqN(0, 3)) * p3d_1;
    Eigen::Matrix<typename A::Scalar, 3, Eigen::Dynamic> p3 = p2d_2.colwise().homogeneous();

    Eigen::Matrix<typename A::Scalar, 3, 1> t = P2.col(3);

    Eigen::Matrix<typename A::Scalar, 3, 1> p3_X3;
    Eigen::Matrix<typename A::Scalar, 3, 1> p3_t3;

    typename A::Scalar scale_sum = 0;

    for (int i = 0; i < N; ++i)
    {
    p3_X3 = p3.col(i).cross(X3.col(i));
    p3_t3 = p3.col(i).cross(t);

    scale_sum -= p3_X3.dot(p3_t3) / p3_t3.dot(p3_t3);
    }

    return scale_sum / N;
}

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

    ls_matrix(Eigen::seqN(2 * m, 2), Eigen::indexing::all) = L * P(Eigen::indexing::all, Eigen::seqN(4 * m, 4));
    }

    p3d_h.col(n) = ls_matrix.jacobiSvd<Eigen::ComputeFullV>().matrixV().col(3); // Previously BDC SVD, Full = Thin
    }

    return p3d_h;
}

// OK
// For N points
template <typename scalar>
struct result_R_t_from_E
{
    Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> P;     // 3x4
    Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> p3d_h; // 4xN
};

// OK
// For N points
// E:     3x3
// p2d_1: 2xN / 3xN
// p2d_2: 2xN
template <typename A, typename B, typename C>
result_R_t_from_E<typename A::Scalar> R_t_from_E(Eigen::MatrixBase<A> const& E, Eigen::MatrixBase<B> const& p2d_1, Eigen::MatrixBase<C> const& p2d_2)
{
    Eigen::Index const N = p2d_1.cols();
    bool const is_3d = p2d_1.rows() > 2;

    Eigen::Matrix<typename A::Scalar, 3, 3> W{ {0, -1, 0}, {1, 0, 0}, {0, 0, 1} };

    Eigen::JacobiSVD<typename A::PlainObject, Eigen::ComputeFullU | Eigen::ComputeFullV> E_svd = E.jacobiSvd<Eigen::ComputeFullU | Eigen::ComputeFullV>(); // Full = Thin

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

    result_R_t_from_E<typename A::Scalar> result;

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> XYZW;
    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> p2d;

    if (is_3d)
    {
    XYZW = p2d_1.colwise().homogeneous();
    }
    else
    {
    p2d = Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic>(4, N);

    p2d(Eigen::seqN(0, 2), Eigen::indexing::all) = p2d_1;
    p2d(Eigen::seqN(2, 2), Eigen::indexing::all) = p2d_2;
    }

    int64_t max_count = -1;
    
    for (int i = 0; i < 4; ++i)
    {
    switch (i)
    {
    case 0: P2 << R1, pt; break;
    case 1: P2 << R1, nt; break;
    case 2: P2 << R2, pt; break;
    case 3: P2 << R2, nt; break;
    }

    if (is_3d)
    {
    P2.col(3) *= compute_scale(p2d_1, p2d_2, P2); // Self assign OK
    }
    else
    {
    PX(Eigen::indexing::all, Eigen::seqN(0, 4)) = P1;
    PX(Eigen::indexing::all, Eigen::seqN(4, 4)) = P2;

    XYZW = triangulate(PX, p2d).colwise().hnormalized().colwise().homogeneous();
    }

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

// OK
// k:      3x1
// return: 3x3
template <typename A>
Eigen::Matrix<typename A::Scalar, 3, 3> matrix_R_cayley(Eigen::MatrixBase<A> const& k)
{
    Eigen::Matrix<typename A::Scalar, 3, 3> R;

    typename A::Scalar kx = k(0);
    typename A::Scalar ky = k(1);
    typename A::Scalar kz = k(2);

    R << 1 + kx * kx - ky * ky - kz * kz, 2 * kx * ky - 2 * kz, 2 * kx * kz + 2 * ky,
         2 * kx * ky + 2 * kz, 1 - kx * kx + ky * ky - kz * kz, 2 * ky * kz - 2 * kx,
         2 * kx * kz - 2 * ky, 2 * ky * kz + 2 * kx, 1 - kx * kx - ky * ky + kz * kz;
    R /= 1 + kx * kx + ky * ky + kz * kz;

    return R;
}

// OK
// v:      3x1
// return: 3x3
template <typename A>
Eigen::Matrix<typename A::Scalar, 3, 3> matrix_cross(Eigen::MatrixBase<A> const& v)
{
    Eigen::Matrix<typename A::Scalar, 3, 3> M;

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

// OK
// E:      3x3
// return: 3x3
template <typename A>
Eigen::Matrix<typename A::Scalar, 3, 3> normalize_E(Eigen::MatrixBase<A> const& E)
{
    Eigen::JacobiSVD<typename A::PlainObject, Eigen::ComputeFullU | Eigen::ComputeFullV> E_svd = E.jacobiSvd<Eigen::ComputeFullU | Eigen::ComputeFullV>();
    return E_svd.matrixU() * Eigen::Matrix<typename A::Scalar, 3, 3>{ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0 } } * E_svd.matrixV().transpose();
}

// OK
// r:      3x1
// return: 3x3
template <typename A>
Eigen::Matrix<typename A::Scalar, 3, 3> matrix_R_rodrigues(Eigen::MatrixBase<A> const& r, typename A::Scalar tolerance = 1e-15) // default
{
    typename A::Scalar n = r.norm();
    if (n < tolerance) { return Eigen::Matrix<typename A::Scalar, 3, 3>::Identity(); }
    return Eigen::AngleAxis<typename A::Scalar>(n, r / n).toRotationMatrix();
}

// OK
// R:      3x3
// return: 3x1
template <typename A>
Eigen::Matrix<typename A::Scalar, 3, 1> vector_r_rodrigues(Eigen::MatrixBase<A> const& R)
{
    Eigen::AngleAxis<typename A::Scalar> aa(R);
    return aa.axis() * aa.angle();
}

// OK
// r: 3x1
// t: 3x1
template <typename A, typename B>
bool is_valid_pose(Eigen::MatrixBase<A> const& r, Eigen::MatrixBase<B> const& t)
{
    typename A::Scalar r_sum = r(0) + r(1) + r(2);
    typename A::Scalar t_sum = t(0) + t(1) + t(2);
    typename A::Scalar x_sum = r_sum + t_sum;

    return std::isfinite(x_sum);
}

// OK
// p2d_1:  2xN
// p2d_2:  2xN
// p2d_3:  2xN
// return: 27x4*N
template <typename A, typename B, typename C>
Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_TFT_constraints(Eigen::MatrixBase<A> const& p2d_1, Eigen::MatrixBase<B> const& p2d_2, Eigen::MatrixBase<C> const& p2d_3)
{
    Eigen::Index const N = p2d_1.cols();

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> Q(27, 4 * N);

    for (int i = 0; i < N; ++i)
    {
    typename A::Scalar x1 = p2d_1(0, i);
    typename A::Scalar y1 = p2d_1(1, i);
    typename A::Scalar x2 = p2d_2(0, i);
    typename A::Scalar y2 = p2d_2(1, i);
    typename A::Scalar x3 = p2d_3(0, i);
    typename A::Scalar y3 = p2d_3(1, i);

    int b = 4 * i;

    int b0 = b + 0;
    int b1 = b + 1;
    int b2 = b + 2;
    int b3 = b + 3;

    Q( 0, b0) = x1;
    Q( 1, b0) = 0;
    Q( 2, b0) = -x1 * x2;
    Q( 3, b0) = 0;
    Q( 4, b0) = 0;
    Q( 5, b0) = 0;
    Q( 6, b0) = -x1 * x3;
    Q( 7, b0) = 0;
    Q( 8, b0) = x1 * x2 * x3;
    Q( 9, b0) = y1;
    Q(10, b0) = 0;
    Q(11, b0) = -x2 * y1;
    Q(12, b0) = 0;
    Q(13, b0) = 0;
    Q(14, b0) = 0;
    Q(15, b0) = -x3 * y1;
    Q(16, b0) = 0;
    Q(17, b0) = x2 * x3 * y1;
    Q(18, b0) = 1;
    Q(19, b0) = 0;
    Q(20, b0) = -x2;
    Q(21, b0) = 0;
    Q(22, b0) = 0;
    Q(23, b0) = 0;
    Q(24, b0) = -x3;
    Q(25, b0) = 0;
    Q(26, b0) = x2 * x3;

    Q( 0, b1) = 0;
    Q( 1, b1) = x1;
    Q( 2, b1) = -x1 * y2;
    Q( 3, b1) = 0;
    Q( 4, b1) = 0;
    Q( 5, b1) = 0;
    Q( 6, b1) = 0;
    Q( 7, b1) = -x1 * x3;
    Q( 8, b1) = x1 * x3 * y2;
    Q( 9, b1) = 0;
    Q(10, b1) = y1;
    Q(11, b1) = -y1 * y2;
    Q(12, b1) = 0;
    Q(13, b1) = 0;
    Q(14, b1) = 0;
    Q(15, b1) = 0;
    Q(16, b1) = -x3 * y1;
    Q(17, b1) = x3 * y1 * y2;
    Q(18, b1) = 0;
    Q(19, b1) = 1;
    Q(20, b1) = -y2;
    Q(21, b1) = 0;
    Q(22, b1) = 0;
    Q(23, b1) = 0;
    Q(24, b1) = 0;
    Q(25, b1) = -x3;
    Q(26, b1) = x3 * y2;

    Q( 0, b2) = 0;
    Q( 1, b2) = 0;
    Q( 2, b2) = 0;
    Q( 3, b2) = x1;
    Q( 4, b2) = 0;
    Q( 5, b2) = -x1 * x2;
    Q( 6, b2) = -x1 * y3;
    Q( 7, b2) = 0;
    Q( 8, b2) = x1 * x2 * y3;
    Q( 9, b2) = 0;
    Q(10, b2) = 0;
    Q(11, b2) = 0;
    Q(12, b2) = y1;
    Q(13, b2) = 0;
    Q(14, b2) = -x2 * y1;
    Q(15, b2) = -y1 * y3;
    Q(16, b2) = 0;
    Q(17, b2) = x2 * y1 * y3;
    Q(18, b2) = 0;
    Q(19, b2) = 0;
    Q(20, b2) = 0;
    Q(21, b2) = 1;
    Q(22, b2) = 0;
    Q(23, b2) = -x2;
    Q(24, b2) = -y3;
    Q(25, b2) = 0;
    Q(26, b2) = x2 * y3;

    Q( 0, b3) = 0;
    Q( 1, b3) = 0;
    Q( 2, b3) = 0;
    Q( 3, b3) = 0;
    Q( 4, b3) = x1;
    Q( 5, b3) = -x1 * y2;
    Q( 6, b3) = 0;
    Q( 7, b3) = -x1 * y3;
    Q( 8, b3) = x1 * y2 * y3;
    Q( 9, b3) = 0;
    Q(10, b3) = 0;
    Q(11, b3) = 0;
    Q(12, b3) = 0;
    Q(13, b3) = y1;
    Q(14, b3) = -y1 * y2;
    Q(15, b3) = 0;
    Q(16, b3) = -y1 * y3;
    Q(17, b3) = y1 * y2 * y3;
    Q(18, b3) = 0;
    Q(19, b3) = 0;
    Q(20, b3) = 0;
    Q(21, b3) = 0;
    Q(22, b3) = 1;
    Q(23, b3) = -y2;
    Q(24, b3) = 0;
    Q(25, b3) = -y3;
    Q(26, b3) = y2 * y3;
    }

    return Q;
}

// OK
// TFT:    27x1
// return: 3x2
template <typename A>
Eigen::Matrix<typename A::Scalar, 3, 2> epipoles_from_TFT(Eigen::MatrixBase<A> const& TFT)
{
    Eigen::Matrix<typename A::Scalar, 3, 3> t1 = TFT(Eigen::seqN( 0, 9)).reshaped(3, 3);
    Eigen::Matrix<typename A::Scalar, 3, 3> t2 = TFT(Eigen::seqN( 9, 9)).reshaped(3, 3);
    Eigen::Matrix<typename A::Scalar, 3, 3> t3 = TFT(Eigen::seqN(18, 9)).reshaped(3, 3);

    Eigen::JacobiSVD<Eigen::Matrix<typename A::Scalar, 3, 3>, Eigen::ComputeFullV | Eigen::ComputeFullU> svd_t1 = t1.jacobiSvd<Eigen::ComputeFullV | Eigen::ComputeFullU>(); // Full = Thin
    Eigen::JacobiSVD<Eigen::Matrix<typename A::Scalar, 3, 3>, Eigen::ComputeFullV | Eigen::ComputeFullU> svd_t2 = t2.jacobiSvd<Eigen::ComputeFullV | Eigen::ComputeFullU>(); // Full = Thin
    Eigen::JacobiSVD<Eigen::Matrix<typename A::Scalar, 3, 3>, Eigen::ComputeFullV | Eigen::ComputeFullU> svd_t3 = t3.jacobiSvd<Eigen::ComputeFullV | Eigen::ComputeFullU>(); // Full = Thin

    Eigen::Matrix<typename A::Scalar, 3, 3> vx;
    Eigen::Matrix<typename A::Scalar, 3, 3> ux;
    Eigen::Matrix<typename A::Scalar, 3, 2> e;

    vx.col(0) =  svd_t1.matrixV().col(2);
    vx.col(1) =  svd_t2.matrixV().col(2);
    vx.col(2) =  svd_t3.matrixV().col(2);

    ux.col(0) = -svd_t1.matrixU().col(2);
    ux.col(1) = -svd_t2.matrixU().col(2);
    ux.col(2) = -svd_t3.matrixU().col(2);

    e.col(0) = -ux.jacobiSvd<Eigen::ComputeFullU>().matrixU().col(2); // Full = Thin
    e.col(1) = -vx.jacobiSvd<Eigen::ComputeFullU>().matrixU().col(2); // Full = Thin

    return e;
}

// OK
template <typename scalar>
struct result_linear_TFT
{
    Eigen::Matrix<scalar, 27, 1> TFT; // 27x1
    Eigen::Matrix<scalar,  3, 4> P2;  //  3x4
    Eigen::Matrix<scalar,  3, 4> P3;  //  3x4
};

// OK
// Q: 27x4*N
template <typename A>
result_linear_TFT<typename A::Scalar> linear_TFT(Eigen::MatrixBase<A> const& Q, typename A::Scalar threshold = 0) // default
{
    Eigen::Matrix<typename A::Scalar, 27, 1> t = -(Q.bdcSvd<Eigen::ComputeThinU>().matrixU().col(26)); // Previously BDC SVD, jacobiSVD drift
    Eigen::Matrix<typename A::Scalar,  3, 2> e = epipoles_from_TFT(t); // OK

    Eigen::Matrix<typename A::Scalar, 3, 3> I3 = Eigen::Matrix<typename A::Scalar, 3, 3>::Identity();
    Eigen::Matrix<typename A::Scalar, 9, 9> I9 = Eigen::Matrix<typename A::Scalar, 9, 9>::Identity();

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> E(27, 18);

    E(Eigen::indexing::all, Eigen::seqN(0, 9)) = Eigen::kroneckerProduct(I3, Eigen::kroneckerProduct(e.col(1), I3));
    E(Eigen::indexing::all, Eigen::seqN(9, 9)) = Eigen::kroneckerProduct(I9,                        -e.col(0));         

    Eigen::BDCSVD<Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic>, Eigen::ComputeThinU | Eigen::ComputeThinV> svd_E = E.bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>(); // BDC SVD gives NaN sometimes, Jacobi SVD doesn't
    if (threshold > 0) { svd_E.setThreshold(threshold); }
    Eigen::Index r = svd_E.rank();

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> Up = svd_E.matrixU()(Eigen::indexing::all, Eigen::seqN(0, r)); // Rank seems to be always 15

    Eigen::Vector<typename A::Scalar, Eigen::Dynamic> tp = ((Q.transpose() * Up).bdcSvd<Eigen::ComputeThinV>().matrixV()(Eigen::indexing::all, Eigen::indexing::last)); // Previously BDC SVD
    Eigen::Vector<typename A::Scalar, Eigen::Dynamic> ap = svd_E.matrixV()(Eigen::indexing::all, Eigen::seqN(0, r)) * svd_E.singularValues()(Eigen::seqN(0, r)).asDiagonal().inverse() * tp;

    result_linear_TFT<typename A::Scalar> result;

    result.TFT = Up * tp;
    result.P2  = (Eigen::Matrix<typename A::Scalar, 3, 4>() << ap(Eigen::seqN(0, 9)).reshaped(3, 3), e.col(0)).finished();
    result.P3  = (Eigen::Matrix<typename A::Scalar, 3, 4>() << ap(Eigen::seqN(9, 9)).reshaped(3, 3), e.col(1)).finished();

    return result;
}

// OK
template <typename scalar>
struct result_R_t_from_TFT
{
    result_R_t_from_E<scalar> v2;
    result_R_t_from_E<scalar> v3;
};

// OK
// TFT:   27x1
// p2d_1: 2xN / 3xN
// p2d_2: 2xN
// p2d_3: 2xN
template <typename A, typename B, typename C, typename D>
result_R_t_from_TFT<typename A::Scalar> R_t_from_TFT(Eigen::MatrixBase<A> const& TFT, Eigen::MatrixBase<B> const& p2d_1, Eigen::MatrixBase<C> const& p2d_2, Eigen::MatrixBase<D> const& p2d_3)
{
    Eigen::Index const N = p2d_1.cols();

    Eigen::Matrix<typename A::Scalar, 3, 2> e = epipoles_from_TFT(TFT); // OK

    Eigen::Matrix<typename A::Scalar, 3, 1> e0 = e.col(0);
    Eigen::Matrix<typename A::Scalar, 3, 1> e1 = e.col(1);

    Eigen::Matrix<typename A::Scalar, 3, 3> epi21_x = matrix_cross(e0); // OK
    Eigen::Matrix<typename A::Scalar, 3, 3> epi31_x = matrix_cross(e1); // OK

    Eigen::Matrix<typename A::Scalar, 3, 3> T1 = TFT(Eigen::seqN( 0, 9)).reshaped(3, 3);
    Eigen::Matrix<typename A::Scalar, 3, 3> T2 = TFT(Eigen::seqN( 9, 9)).reshaped(3, 3);
    Eigen::Matrix<typename A::Scalar, 3, 3> T3 = TFT(Eigen::seqN(18, 9)).reshaped(3, 3);

    Eigen::Matrix<typename A::Scalar, 3, 3> D21;
    Eigen::Matrix<typename A::Scalar, 3, 3> D31;

    D21.col(0) = T1             * e1;
    D21.col(1) = T2             * e1;
    D21.col(2) = T3             * e1;

    D31.col(0) = T1.transpose() * e0;
    D31.col(1) = T2.transpose() * e0;
    D31.col(2) = T3.transpose() * e0;

    Eigen::Matrix<typename A::Scalar, 3, 3> E21 = epi21_x * D21;
    Eigen::Matrix<typename A::Scalar, 3, 3> E31 = epi31_x * D31;

    result_R_t_from_TFT<typename A::Scalar> result;

    result.v2 = R_t_from_E(E21, p2d_1,                                   p2d_2); // OK
    result.v3 = R_t_from_E(E31, result.v2.p3d_h.colwise().hnormalized(), p2d_3); // OK

    return result;
}

// OK
template <typename scalar>
struct result_normalize_points
{
    Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> H; // 3x3
    Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> p; // 2xN
};

// OK
// p_i: 2xN
template <typename A>
result_normalize_points<typename A::Scalar> normalize_points(Eigen::MatrixBase<A> const& p_i)
{
    typename A::Scalar const sqrt_2 = static_cast<typename A::Scalar>(1.4142135623730950488016887242097);

    Eigen::Matrix<typename A::Scalar, 2, 1> p0 = p_i.rowwise().mean();

    typename A::Scalar s = sqrt_2 / (p_i.colwise() - p0).colwise().norm().mean();

    result_normalize_points<typename A::Scalar> result;

    result.H = Eigen::Matrix<typename A::Scalar, 3, 3>
    {
        {s, 0, -s * p0(0)},
        {0, s, -s * p0(1)},
        {0, 0,          1},
    };

    result.p = result.H(Eigen::seqN(0, 2), Eigen::indexing::all) * p_i.colwise().homogeneous();

    return result;
}

// OK
// tft: 27x1
// M1:   3x3
// M2:   3x3
// M3:   3x3
template <typename A, typename B, typename C, typename D>
Eigen::Matrix<typename A::Scalar, 27, 1> transform_TFT(Eigen::MatrixBase<A> const& tft, Eigen::MatrixBase<B> const& M1, Eigen::MatrixBase<C> const& M2, Eigen::MatrixBase<D> const& M3, bool inverse)
{
    Eigen::Matrix<typename A::Scalar, 3, 3> t1 = tft(Eigen::seqN( 0, 9)).reshaped(3, 3);
    Eigen::Matrix<typename A::Scalar, 3, 3> t2 = tft(Eigen::seqN( 9, 9)).reshaped(3, 3);
    Eigen::Matrix<typename A::Scalar, 3, 3> t3 = tft(Eigen::seqN(18, 9)).reshaped(3, 3);

    Eigen::Matrix<typename A::Scalar, 27, 1> t;

    if (!inverse)
    {
    Eigen::Matrix<typename A::Scalar, 3, 3> M1i = M1.inverse();

    t(Eigen::seqN( 0, 9)) = (M2 * (M1i(0, 0) * t1 + M1i(1, 0) * t2 + M1i(2, 0) * t3) * M3.transpose()).reshaped(9, 1);
    t(Eigen::seqN( 9, 9)) = (M2 * (M1i(0, 1) * t1 + M1i(1, 1) * t2 + M1i(2, 1) * t3) * M3.transpose()).reshaped(9, 1);
    t(Eigen::seqN(18, 9)) = (M2 * (M1i(0, 2) * t1 + M1i(1, 2) * t2 + M1i(2, 2) * t3) * M3.transpose()).reshaped(9, 1);
    }
    else
    {
    Eigen::Matrix<typename A::Scalar, 3, 3> M2i = M2.inverse();
    Eigen::Matrix<typename A::Scalar, 3, 3> M3i = M3.inverse();

    t(Eigen::seqN( 0, 9)) = (M2i * (M1(0, 0) * t1 + M1(1, 0) * t2 + M1(2, 0) * t3) * M3i.transpose()).reshaped(9, 1);
    t(Eigen::seqN( 9, 9)) = (M2i * (M1(0, 1) * t1 + M1(1, 1) * t2 + M1(2, 1) * t3) * M3i.transpose()).reshaped(9, 1);
    t(Eigen::seqN(18, 9)) = (M2i * (M1(0, 2) * t1 + M1(1, 2) * t2 + M1(2, 2) * t3) * M3i.transpose()).reshaped(9, 1);
    }

    return t.normalized();
}

// OK
// r_gt: 3x1
// t_gt: 3x1
// r:    3x1
// t:    3x1
template <typename A, typename B, typename C, typename D>
Eigen::Matrix<typename A::Scalar, 2, 1> compute_error(Eigen::MatrixBase<A> const& r_gt, Eigen::MatrixBase<B> const& t_gt, Eigen::MatrixBase<C> const& r, Eigen::MatrixBase<D> const& t)
{
    typename A::Scalar r_error = vector_r_rodrigues(matrix_R_rodrigues(r_gt).transpose() * matrix_R_rodrigues(r)).norm();
    typename A::Scalar t_error = (t_gt - t).norm();

    return Eigen::Matrix<typename A::Scalar, 2, 1>{ r_error, t_error };
}




