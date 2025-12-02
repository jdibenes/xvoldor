
#include <thread>
#include <iostream>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h>
#include "helpers_eigen.h"

//-----------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------

// OK
struct
result_normalize_points
{
    Eigen::Matrix<float, 3, 3> H;
    Eigen::MatrixXf p; // 2xN
};

// OK
static
result_normalize_points
normalize_points
(
    Eigen::Ref<const Eigen::MatrixXf> const& p_i // 2xN
)
{
    Eigen::Matrix<float, 2, 1> p0 = p_i.rowwise().mean();
    float s = std::sqrt(2.0f) / (p_i.colwise() - p0).colwise().norm().mean();

    result_normalize_points result;

    result.H = Eigen::Matrix<float, 3, 3>
    {
        {   s, 0.0f, -s * p0(0)},
        {0.0f,    s, -s * p0(1)},
        {0.0f, 0.0f,       1.0f},
    };

    result.p = result.H(Eigen::seqN(0, 2), Eigen::all) * p_i.colwise().homogeneous();

    return result;
}

// OK
static
Eigen::Matrix<float, 3, 3>
cross_matrix
(
    Eigen::Ref<const Eigen::Matrix<float, 3, 1>> const& v
)
{
    Eigen::Matrix<float, 3, 3> M;

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

//-----------------------------------------------------------------------------
// M-View Geometry
//-----------------------------------------------------------------------------

// OK
static
Eigen::MatrixXf // 4xN
triangulate
(
    Eigen::Ref<const Eigen::MatrixXf> const& P,  // 3x4*M
    Eigen::Ref<const Eigen::MatrixXf> const& p2d // 2*MxN
)
{
    int const M = P.cols() / 4;
    int const N = p2d.cols();

    Eigen::Matrix<float, 2, 3> L
    {
        {0.0f, -1.0f, 0.0f},
        {1.0f,  0.0f, 0.0f},
    };

    Eigen::MatrixXf ls_matrix(2 * M, 4);
    Eigen::MatrixXf p3d_h(4, N);

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

//-----------------------------------------------------------------------------
// 2-View Geometry
//-----------------------------------------------------------------------------

// OK
static
float
compute_scale
(
    Eigen::Ref<const Eigen::MatrixXf> const& p2d, // 2xN
    Eigen::Ref<const Eigen::MatrixXf> const& p3d, // 3xN
    Eigen::Ref<const Eigen::Matrix<float, 3, 4>> const& P2
)
{
    int const N = p2d.cols();

    Eigen::Matrix<float, 3, Eigen::Dynamic> X3 = P2(Eigen::all, Eigen::seqN(0, 3)) * p3d;
    Eigen::Matrix<float, 3, Eigen::Dynamic> p3 = p2d.colwise().homogeneous();
    Eigen::Matrix<float, 3, 1> p3_X3;
    Eigen::Matrix<float, 3, 1> p3_t3;

    float num = 0;
    float den = 0;

    for (int i = 0; i < N; ++i)
    {
        p3_X3 = p3.col(i).cross(X3.col(i));
        p3_t3 = p3.col(i).cross(P2.col(3));

        num -= p3_X3.dot(p3_t3);
        den += p3_t3.dot(p3_t3);
    }

    return num / den; // TODO: Mean or Median or ... ?
}

// OK
static
float
compute_scale_los
(
    Eigen::Ref<const Eigen::MatrixXf> const& p2d, // 2xN
    Eigen::Ref<const Eigen::MatrixXf> const& p3d, // 3xN
    Eigen::Ref<const Eigen::Matrix<float, 3, 4>> const& P2
)
{
    int const N = p2d.cols();

    Eigen::Matrix<float, 3, Eigen::Dynamic> X3 = P2(Eigen::all, Eigen::seqN(0, 3)) * p3d;
    Eigen::Matrix<float, 3, Eigen::Dynamic> p3 = p2d.colwise().homogeneous();

    float s = 0;
    for (int i = 0; i < N; ++i) { s += p3.col(i).cross(X3.col(i)).norm() / p3.col(i).cross(P2.col(3)).norm(); }
    return s / N; // TODO: Mean or Median or ... ?
}

// OK
struct
result_R_t_from_E
{
    Eigen::Matrix<float, 3, 4> P;
    Eigen::MatrixXf p3d_h;
};

// OK
static
result_R_t_from_E
R_t_from_E
(
    Eigen::Ref<const Eigen::Matrix<float, 3, 3>> const& E,
    Eigen::Ref<const Eigen::MatrixXf> const& p2d_1, // 2xN
    Eigen::Ref<const Eigen::MatrixXf> const& p2d_2, // 2xN
    Eigen::Ref<const Eigen::MatrixXf> const& p3d_1, // 3xN
    bool use_prior
)
{
    int const N = p2d_1.cols();

    Eigen::MatrixXf p2d(4, N);

    p2d(Eigen::seqN(0, 2), Eigen::all) = p2d_1;
    p2d(Eigen::seqN(2, 2), Eigen::all) = p2d_2;

    // TODO: Constant
    Eigen::Matrix<float, 3, 3> W
    {
        {0.0f, -1.0f, 0.0f},
        {1.0f,  0.0f, 0.0f},
        {0.0f,  0.0f, 1.0f},
    };

    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> E_svd = E.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV); // Full = Thin

    Eigen::Matrix<float, 3, 3> U  = E_svd.matrixU();
    Eigen::Matrix<float, 3, 3> Vt = E_svd.matrixV().transpose();

    Eigen::Matrix<float, 3, 3> R1 = U * W             * Vt;
    Eigen::Matrix<float, 3, 3> R2 = U * W.transpose() * Vt;

    if (R1.determinant() < 0) { R1 = -R1; } // Self assign
    if (R2.determinant() < 0) { R2 = -R2; } // Self assign

    Eigen::Matrix<float, 3, 1> pt = U.col(2);
    Eigen::Matrix<float, 3, 1> nt = -pt;

    Eigen::Matrix<float, 3, 4> P1 = Eigen::Matrix<float, 3, 4>::Identity(); // TODO: constant
    Eigen::Matrix<float, 3, 4> P2;
    Eigen::Matrix<float, 3, 8> PX;

    result_R_t_from_E result;
    Eigen::MatrixXf XYZW;
    int64_t max_count = -1;

    if (use_prior)
    { 
        XYZW = p3d_1.colwise().homogeneous();
        result.p3d_h = XYZW;
    }

    for (int i = 0; i < 4; ++i)
    {
        switch (i)
        {
        case 0:
            P2(Eigen::all, Eigen::seqN(0, 3)) = R1;
            P2.col(3) = pt;
            break;
        case 1:
            P2(Eigen::all, Eigen::seqN(0, 3)) = R1;
            P2.col(3) = nt;
            break;
        case 2:
            P2(Eigen::all, Eigen::seqN(0, 3)) = R2;
            P2.col(3) = pt;
            break;
        case 3:
            P2(Eigen::all, Eigen::seqN(0, 3)) = R2;
            P2.col(3) = nt;
            break;
        }

        if (!use_prior) 
        {
            PX(Eigen::all, Eigen::seqN(0, 4)) = P1;
            PX(Eigen::all, Eigen::seqN(4, 4)) = P2;

            XYZW = triangulate(PX, p2d).colwise().hnormalized().colwise().homogeneous(); // OK
        }
        else
        {
            P2.col(3) = compute_scale(p2d_2, p3d_1, P2) * P2.col(3); // Self assign // OK
        }

        int64_t count = ((P1 * XYZW).row(2).array() > 0).count() + ((P2 * XYZW).row(2).array() > 0).count();
        if (count < max_count) { continue; }
        max_count = count;

        result.P = P2;
        if (!use_prior) { result.p3d_h = XYZW; }
    }

    return result;
}

//-----------------------------------------------------------------------------
// 3-View Geometry
//-----------------------------------------------------------------------------

// OK
static
Eigen::MatrixXf // 27x4*N
build_A
(
    Eigen::Ref<const Eigen::MatrixXf> const& p2d_1, // 2xN
    Eigen::Ref<const Eigen::MatrixXf> const& p2d_2, // 2xN
    Eigen::Ref<const Eigen::MatrixXf> const& p2d_3  // 2xN
)
{
    int const N = p2d_1.cols();

    Eigen::MatrixXf A(27, 4 * N);

    for (int i = 0; i < N; ++i)
    {
        float x1 = p2d_1(0, i);
        float y1 = p2d_1(1, i);
        float x2 = p2d_2(0, i);
        float y2 = p2d_2(1, i);
        float x3 = p2d_3(0, i);
        float y3 = p2d_3(1, i);

        int b = 4 * i;

        int b0 = b + 0;
        int b1 = b + 1;
        int b2 = b + 2;
        int b3 = b + 3;

        A( 0, b0) = x1;
        A( 1, b0) = 0;
        A( 2, b0) = -x1 * x2;
        A( 3, b0) = 0;
        A( 4, b0) = 0;
        A( 5, b0) = 0;
        A( 6, b0) = -x1 * x3;
        A( 7, b0) = 0;
        A( 8, b0) = x1 * x2 * x3;
        A( 9, b0) = y1;
        A(10, b0) = 0;
        A(11, b0) = -x2 * y1;
        A(12, b0) = 0;
        A(13, b0) = 0;
        A(14, b0) = 0;
        A(15, b0) = -x3 * y1;
        A(16, b0) = 0;
        A(17, b0) = x2 * x3 * y1;
        A(18, b0) = 1;
        A(19, b0) = 0;
        A(20, b0) = -x2;
        A(21, b0) = 0;
        A(22, b0) = 0;
        A(23, b0) = 0;
        A(24, b0) = -x3;
        A(25, b0) = 0;
        A(26, b0) = x2 * x3;

        A( 0, b1) = 0;
        A( 1, b1) = x1;
        A( 2, b1) = -x1 * y2;
        A( 3, b1) = 0;
        A( 4, b1) = 0;
        A( 5, b1) = 0;
        A( 6, b1) = 0;
        A( 7, b1) = -x1 * x3;
        A( 8, b1) = x1 * x3 * y2;
        A( 9, b1) = 0;
        A(10, b1) = y1;
        A(11, b1) = -y1 * y2;
        A(12, b1) = 0;
        A(13, b1) = 0;
        A(14, b1) = 0;
        A(15, b1) = 0;
        A(16, b1) = -x3 * y1;
        A(17, b1) = x3 * y1 * y2;
        A(18, b1) = 0;
        A(19, b1) = 1;
        A(20, b1) = -y2;
        A(21, b1) = 0;
        A(22, b1) = 0;
        A(23, b1) = 0;
        A(24, b1) = 0;
        A(25, b1) = -x3;
        A(26, b1) = x3 * y2;

        A( 0, b2) = 0;
        A( 1, b2) = 0;
        A( 2, b2) = 0;
        A( 3, b2) = x1;
        A( 4, b2) = 0;
        A( 5, b2) = -x1 * x2;
        A( 6, b2) = -x1 * y3;
        A( 7, b2) = 0;
        A( 8, b2) = x1 * x2 * y3;
        A( 9, b2) = 0;
        A(10, b2) = 0;
        A(11, b2) = 0;
        A(12, b2) = y1;
        A(13, b2) = 0;
        A(14, b2) = -x2 * y1;
        A(15, b2) = -y1 * y3;
        A(16, b2) = 0;
        A(17, b2) = x2 * y1 * y3;
        A(18, b2) = 0;
        A(19, b2) = 0;
        A(20, b2) = 0;
        A(21, b2) = 1;
        A(22, b2) = 0;
        A(23, b2) = -x2;
        A(24, b2) = -y3;
        A(25, b2) = 0;
        A(26, b2) = x2 * y3;

        A( 0, b3) = 0;
        A( 1, b3) = 0;
        A( 2, b3) = 0;
        A( 3, b3) = 0;
        A( 4, b3) = x1;
        A( 5, b3) = -x1 * y2;
        A( 6, b3) = 0;
        A( 7, b3) = -x1 * y3;
        A( 8, b3) = x1 * y2 * y3;
        A( 9, b3) = 0;
        A(10, b3) = 0;
        A(11, b3) = 0;
        A(12, b3) = 0;
        A(13, b3) = y1;
        A(14, b3) = -y1 * y2;
        A(15, b3) = 0;
        A(16, b3) = -y1 * y3;
        A(17, b3) = y1 * y2 * y3;
        A(18, b3) = 0;
        A(19, b3) = 0;
        A(20, b3) = 0;
        A(21, b3) = 0;
        A(22, b3) = 1;
        A(23, b3) = -y2;
        A(24, b3) = 0;
        A(25, b3) = -y3;
        A(26, b3) = y2 * y3;
    }

    return A;
}

// OK
static 
Eigen::Matrix<float, 3, 2>
epipoles_from_TFT
(
    Eigen::Ref<const Eigen::Matrix<float, 27, 1>> const& TFT
)
{
    Eigen::Matrix<float, 3, 3> t1 = TFT(Eigen::seqN( 0, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> t2 = TFT(Eigen::seqN( 9, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> t3 = TFT(Eigen::seqN(18, 9)).reshaped(3, 3);

    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd_t1 = t1.jacobiSvd(Eigen::ComputeFullV | Eigen::ComputeFullU); // Full = Thin
    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd_t2 = t2.jacobiSvd(Eigen::ComputeFullV | Eigen::ComputeFullU); // Full = Thin
    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd_t3 = t3.jacobiSvd(Eigen::ComputeFullV | Eigen::ComputeFullU); // Full = Thin

    Eigen::Matrix<float, 3, 3> vx;
    Eigen::Matrix<float, 3, 3> ux;
    Eigen::Matrix<float, 3, 2> e;

    vx.col(0) =  svd_t1.matrixV().col(2);
    vx.col(1) =  svd_t2.matrixV().col(2);
    vx.col(2) =  svd_t3.matrixV().col(2);

    ux.col(0) = -svd_t1.matrixU().col(2);
    ux.col(1) = -svd_t2.matrixU().col(2);
    ux.col(2) = -svd_t3.matrixU().col(2);

    e.col(0) = -ux.jacobiSvd(Eigen::ComputeFullU).matrixU().col(2); // Full = Thin
    e.col(1) = -vx.jacobiSvd(Eigen::ComputeFullU).matrixU().col(2); // Full = Thin

    return e;
}

// OK
struct
result_linear_TFT
{
    Eigen::Matrix<float, 27, 1> TFT;
    Eigen::Matrix<float,  3, 4> P2;
    Eigen::Matrix<float,  3, 4> P3;
};

// OK
static
result_linear_TFT
linear_TFT
(
    Eigen::Ref<const Eigen::MatrixXf> const& A, // 27x4*N
    float threshold = 0
)
{
    Eigen::Matrix<float, 27, 1> t = -(A.bdcSvd(Eigen::ComputeThinU).matrixU().col(26)); // Previously BDC SVD, jacobiSVD drift
    Eigen::Matrix<float,  3, 2> e = epipoles_from_TFT(t); // OK

    Eigen::Matrix<float, 3, 3> I3 = Eigen::Matrix<float, 3, 3>::Identity(); // TODO: constant
    Eigen::Matrix<float, 9, 9> I9 = Eigen::Matrix<float, 9, 9>::Identity(); // TODO: constant

    Eigen::MatrixXf E(27, 18);

    E(Eigen::all, Eigen::seqN(0, 9)) = Eigen::kroneckerProduct(I3, Eigen::kroneckerProduct(e.col(1), I3));
    E(Eigen::all, Eigen::seqN(9, 9)) = Eigen::kroneckerProduct(I9,                        -e.col(0));         

    Eigen::BDCSVD<Eigen::MatrixXf> svd_E = E.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);  // BDC SVD gives NaN sometimes, Jacobi SVD doesn't
    if (threshold > 0) { svd_E.setThreshold(threshold); }
    int r = svd_E.rank();

    Eigen::MatrixXf Up = svd_E.matrixU()(Eigen::all, Eigen::seqN(0, r)); // Rank seems to be always 15
    Eigen::VectorXf tp = ((A.transpose() * Up).bdcSvd(Eigen::ComputeThinV).matrixV()(Eigen::all, Eigen::last)); // Previously BDC SVD
    Eigen::VectorXf ap = svd_E.matrixV()(Eigen::all, Eigen::seqN(0, r)) * svd_E.singularValues()(Eigen::seqN(0, r)).asDiagonal().inverse() * tp;

    result_linear_TFT result;

    result.TFT = Up * tp;

    result.P2(Eigen::all, Eigen::seqN(0, 3)) = ap(Eigen::seqN(0, 9)).reshaped(3, 3);
    result.P2.col(3) = e.col(0);
    result.P3(Eigen::all, Eigen::seqN(0, 3)) = ap(Eigen::seqN(9, 9)).reshaped(3, 3);
    result.P3.col(3) = e.col(1);

    return result;
}

// OK
struct
result_R_t_from_TFT
{
    result_R_t_from_E v2;
    result_R_t_from_E v3;
};

// OK
static 
result_R_t_from_TFT
R_t_from_TFT
(
    Eigen::Ref<const Eigen::Matrix<float, 27, 1>> const& TFT,
    Eigen::Ref<const Eigen::MatrixXf> const& p2d_1, // 2xN
    Eigen::Ref<const Eigen::MatrixXf> const& p2d_2, // 2xN
    Eigen::Ref<const Eigen::MatrixXf> const& p2d_3, // 2xN
    Eigen::Ref<const Eigen::MatrixXf> const& p3d_1, // 3xN
    bool use_prior
)
{
    int const N = p2d_1.cols();

    Eigen::Matrix<float, 3, 2> e = epipoles_from_TFT(TFT); // OK

    Eigen::Matrix<float, 3, 3> epi21_x = cross_matrix(e.col(0)); // OK
    Eigen::Matrix<float, 3, 3> epi31_x = cross_matrix(e.col(1)); // OK

    Eigen::Matrix<float, 3, 3> T1 = TFT(Eigen::seqN( 0, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> T2 = TFT(Eigen::seqN( 9, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> T3 = TFT(Eigen::seqN(18, 9)).reshaped(3, 3);

    Eigen::Matrix<float, 3, 3> D21;
    Eigen::Matrix<float, 3, 3> D31;

    D21.col(0) = T1 * e.col(1);
    D21.col(1) = T2 * e.col(1);
    D21.col(2) = T3 * e.col(1);

    D31.col(0) = T1.transpose() * e.col(0);
    D31.col(1) = T2.transpose() * e.col(0);
    D31.col(2) = T3.transpose() * e.col(0);

    Eigen::Matrix<float, 3, 3> E21 = epi21_x * D21;
    Eigen::Matrix<float, 3, 3> E31 = epi31_x * D31;

    result_R_t_from_TFT result;

    result.v2 = R_t_from_E(E21, p2d_1, p2d_2, p3d_1, use_prior); // OK
    result.v3 = R_t_from_E(E31, p2d_1, p2d_3, p3d_1, use_prior); // OK

    Eigen::MatrixXf p3d = result.v2.p3d_h.colwise().hnormalized();

    result.v3.P.col(3) = compute_scale(p2d_3, p3d, result.v3.P) * result.v3.P.col(3); // Self assign // OK    

    return result;
}

// OK
static
Eigen::Matrix<float, 27, 1>
transform_TFT
(
    Eigen::Ref<const Eigen::Matrix<float, 27, 1>> const& tft_i,
    Eigen::Ref<const Eigen::Matrix<float, 3, 3>> const& M1,
    Eigen::Ref<const Eigen::Matrix<float, 3, 3>> const& M2,
    Eigen::Ref<const Eigen::Matrix<float, 3, 3>> const& M3,
    bool inverse
)
{
    Eigen::Matrix<float, 3, 3> t1_i = tft_i(Eigen::seqN( 0, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> t2_i = tft_i(Eigen::seqN( 9, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> t3_i = tft_i(Eigen::seqN(18, 9)).reshaped(3, 3);

    Eigen::Matrix<float, 27, 1> t_o;

    if (!inverse)
    {
        Eigen::Matrix<float, 3, 3> M1i = M1.inverse();
        t_o(Eigen::seqN( 0, 9)) = (M2 * (M1i(0, 0) * t1_i + M1i(1, 0) * t2_i + M1i(2, 0) * t3_i) * M3.transpose()).reshaped(9, 1);
        t_o(Eigen::seqN( 9, 9)) = (M2 * (M1i(0, 1) * t1_i + M1i(1, 1) * t2_i + M1i(2, 1) * t3_i) * M3.transpose()).reshaped(9, 1);
        t_o(Eigen::seqN(18, 9)) = (M2 * (M1i(0, 2) * t1_i + M1i(1, 2) * t2_i + M1i(2, 2) * t3_i) * M3.transpose()).reshaped(9, 1);
    }
    else
    {
        Eigen::Matrix<float, 3, 3> M2i = M2.inverse();
        Eigen::Matrix<float, 3, 3> M3i = M3.inverse();
        t_o(Eigen::seqN( 0, 9)) = (M2i * (M1(0, 0) * t1_i + M1(1, 0) * t2_i + M1(2, 0) * t3_i) * M3i.transpose()).reshaped(9, 1);
        t_o(Eigen::seqN( 9, 9)) = (M2i * (M1(0, 1) * t1_i + M1(1, 1) * t2_i + M1(2, 1) * t3_i) * M3i.transpose()).reshaped(9, 1);
        t_o(Eigen::seqN(18, 9)) = (M2i * (M1(0, 2) * t1_i + M1(1, 2) * t2_i + M1(2, 2) * t3_i) * M3i.transpose()).reshaped(9, 1);
    }

    return t_o.normalized();
}

//-----------------------------------------------------------------------------
// Solvers
//-----------------------------------------------------------------------------

// OK
bool
trifocal_R_t_linear
(
    float const* p2d_1,
    float const* p2d_2,
    float const* p2d_3,
    float const* p3d_1,
    int N,
    bool use_prior,
    float* r1,
    float* t1,
    float* r2,
    float* t2
)
{
    Eigen::MatrixXf p2d_1_w = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2d_1, 2, N);
    Eigen::MatrixXf p2d_2_w = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2d_2, 2, N);
    Eigen::MatrixXf p2d_3_w = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p2d_3, 2, N);
    Eigen::MatrixXf p3d_s_w = matrix_from_buffer<float, Eigen::Dynamic, Eigen::Dynamic>(p3d_1, 3, N);

    result_normalize_points rnp_1 = normalize_points(p2d_1_w);
    result_normalize_points rnp_2 = normalize_points(p2d_2_w);
    result_normalize_points rnp_3 = normalize_points(p2d_3_w);

    Eigen::MatrixXf A = build_A(rnp_1.p, rnp_2.p, rnp_3.p); // OK
    result_linear_TFT ltr = linear_TFT(A); // OK
    ltr.TFT = transform_TFT(ltr.TFT, rnp_1.H, rnp_2.H, rnp_3.H, true);

    result_R_t_from_TFT rpt = R_t_from_TFT(ltr.TFT, p2d_1_w, p2d_2_w, p2d_3_w, p3d_s_w, use_prior); // OK
    float world_scale = compute_scale(p2d_2_w, p3d_s_w, rpt.v2.P);

    Eigen::Matrix<float, 3, 3> R12 = rpt.v2.P(Eigen::all, Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 3> R13 = rpt.v3.P(Eigen::all, Eigen::seqN(0, 3));

    Eigen::Matrix<float, 3, 1> t12 = world_scale * rpt.v2.P.col(3);
    Eigen::Matrix<float, 3, 1> t13 = world_scale * rpt.v3.P.col(3);

    Eigen::Matrix<float, 3, 3> R23 = R12.transpose() * R13;
    Eigen::Matrix<float, 3, 1> t23 = R12.transpose() * (t13 - t12);

    Eigen::AngleAxis<float> aa_12(R12);
    Eigen::AngleAxis<float> aa_23(R23);

    Eigen::Matrix<float, 3, 1> r12 = aa_12.angle() * aa_12.axis();
    Eigen::Matrix<float, 3, 1> r23 = aa_23.angle() * aa_23.axis();

    matrix_to_buffer(r12, r1);
    matrix_to_buffer(r23, r2);
    matrix_to_buffer(t12, t1);
    matrix_to_buffer(t23, t2);

    return std::isfinite(r1[0] + r1[1] + r1[2] + t1[0] + t1[1] + t1[2] + r2[0] + r2[1] + r2[2] + t2[0] + t2[1] + t2[2]);
}





//memcpy(r1, r01.data(), 3 * sizeof(float));
//memcpy(r2, r02.data(), 3 * sizeof(float));
//memcpy(t1, t01.data(), 3 * sizeof(float));
//memcpy(t2, t02.data(), 3 * sizeof(float));
//Eigen::MatrixXf p2d_s_w(2, N);
//Eigen::MatrixXf p3d_s_w(3, N);



//memcpy(p2d_1_w.data(), p2d_1, 2 * N * sizeof(float));
//memcpy(p2d_2_w.data(), p2d_2, 2 * N * sizeof(float));
//memcpy(p2d_3_w.data(), p2d_3, 2 * N * sizeof(float));
//memcpy(p2d_s_w.data(), p2d_s, 2 * N * sizeof(float));
//memcpy(p3d_s_w.data(), p3d_s, 3 * N * sizeof(float));






// OK
struct
gh_evaluation
{
    Eigen::MatrixXf f;
    Eigen::MatrixXf g;
    Eigen::MatrixXf A;
    Eigen::MatrixXf B;
    Eigen::MatrixXf C;
    Eigen::MatrixXf D;
};

// OK
struct
gh_result
{
    Eigen::MatrixXf x_opt;
    Eigen::MatrixXf t_opt;
    Eigen::MatrixXf y_opt;
    int iter;
};

// OK
typedef 
gh_evaluation
(*gh_function)
(
    Eigen::Ref<const Eigen::MatrixXf> const& x,
    Eigen::Ref<const Eigen::MatrixXf> const& t,
    Eigen::Ref<const Eigen::MatrixXf> const& y,
    void* user
);

// OK
gh_evaluation
gh_ressl
(
    Eigen::Ref<const Eigen::MatrixXf> const& x_in, // (6*N)x1
    Eigen::Ref<const Eigen::MatrixXf> const& p_in, // 20x1
    Eigen::Ref<const Eigen::MatrixXf> const&,
    void* user
)
{
    // Ind2=1:3;
    // Ind2=Ind2(Ind2~=Ind);
    int Ind = *(Eigen::Index*)user;
    int Ind2[2];
    switch (Ind)
    {
    case 0: Ind2[0] = 1; Ind2[1] = 2; break;
    case 1: Ind2[0] = 0; Ind2[1] = 2; break;
    case 2: Ind2[0] = 0; Ind2[1] = 1; break;
    default:                          break; // TODO: BUG
    }

    // S=reshape(p(1:9),3,3);
    Eigen::Matrix<float, 3, 3> S = p_in(Eigen::seq(0, 8), 0).reshaped(3, 3);

    // e21=ones(3,1);
    // e21(Ind2)=p(10:11);
    // e31=p(18:20);
    Eigen::Matrix<float, 3, 1> e21 = Eigen::Matrix<float, 3, 1>::Ones();
    e21(Ind2) = p_in(Eigen::seq(9, 10), 0);
    Eigen::Matrix<float, 3, 1> e31 = p_in(Eigen::seq(17, 19), 0);

    // mn=zeros(3);
    // mn(:,Ind2)=reshape(p(12:17),3,2);
    Eigen::Matrix<float, 3, 3> mn = Eigen::Matrix<float, 3, 3>::Zero();
    mn(Eigen::all, Ind2) = p_in(Eigen::seq(11, 16), 0).reshaped(3, 2);

    // % tensor
    // T1=(S(:,1)*e21.'+e31*mn(1,:)).';
    Eigen::Matrix<float, 3, 3> T1 = (S(Eigen::all, 0) * e21.transpose() + e31 * mn(0, Eigen::all)).transpose();
    
    // T2=(S(:,2)*e21.'+e31*mn(2,:)).';
    Eigen::Matrix<float, 3, 3> T2 = (S(Eigen::all, 1) * e21.transpose() + e31 * mn(1, Eigen::all)).transpose();
    
    // T3=(S(:,3)*e21.'+e31*mn(3,:)).';
    Eigen::Matrix<float, 3, 3> T3 = (S(Eigen::all, 2) * e21.transpose() + e31 * mn(2, Eigen::all)).transpose();

    // % other slices of the TFT
    // J3=[T1(3,:);T2(3,:);T3(3,:)];
    Eigen::Matrix<float, 3, 3> J3;
    J3 << T1(2, Eigen::all), T2(2, Eigen::all), T3(2, Eigen::all);

    // K3=[T1(:,3),T2(:,3),T3(:,3)];
    Eigen::Matrix<float, 3, 3> K3;    
    K3 << T1(Eigen::all, 2), T2(Eigen::all, 2), T3(Eigen::all, 2);

    // N=size(x,1)/6;
    int N = x_in.rows() / 6;

    // g=[sum(e31.^2)-1;sum(S(:).^2)-1];
    Eigen::MatrixXf g(2, 1);
    g << (e31.squaredNorm() - 1), (S.squaredNorm() - 1);

    // % g jacobian w.r.t. p evaluated in p
    // C=zeros(2,20);
    Eigen::MatrixXf C = Eigen::MatrixXf::Zero(2, 20);

    // C(1,18:20)=2*e31.';
    C(0, Eigen::seq(17, 19)) = 2 * e31.transpose();

    // C(2,1:9)=2*S(:).';
    C(1, Eigen::seq(0, 8)) = 2 * S.reshaped(9, 1).transpose();

    // f=zeros(4*N,1);     % constraints for tensor and observations (trilinearities)
    Eigen::MatrixXf f = Eigen::MatrixXf::Zero(4 * N, 1);

    // Ap=zeros(4*N,27);   % jacobian of f w.r.t. the tensor T
    Eigen::MatrixXf Ap = Eigen::MatrixXf::Zero(4 * N, 27);

    // B=zeros(4*N,6*N);   % jacobian of f w.r.t. the observations
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(4 * N, 6 * N);

    Eigen::Matrix<float, 2, 1> x1;
    Eigen::Matrix<float, 2, 1> x2;
    Eigen::Matrix<float, 2, 1> x3;

    Eigen::Matrix<float, 3, 2> S2{ {0, -1}, {-1, 0}, {0, 0} };
    Eigen::Matrix<float, 3, 2> S3{ {0, -1}, {-1, 0}, {0, 0} };

    Eigen::Matrix<float, 2, 2> tmp_22_swp{ {0, 1}, {1, 0} };
    Eigen::Matrix<float, 2, 1> tmm;

    Eigen::Matrix<float, 3, 3> I3 = Eigen::Matrix<float, 3, 3>::Identity();
    Eigen::Matrix<float, 2, 2> I2 = Eigen::Matrix<float, 2, 2>::Identity();
    Eigen::Matrix<float, 9, 9> I9 = Eigen::Matrix<float, 9, 9>::Identity();

    // for i=1:N
    for (int i = 0; i < N; ++i)
    {
        // % points in the three images for correspondance i
        // ind=6*(i-1);
        int ind = 6 * i;

        // x1=x(ind+1:ind+2);
        x1 = x_in(Eigen::seq(ind + 0, ind + 1), 0);

        // x2=x(ind+3:ind+4);
        x2 = x_in(Eigen::seq(ind + 2, ind + 3), 0);

        // x3=x(ind+5:ind+6);
        x3 = x_in(Eigen::seq(ind + 4, ind + 5), 0);

        // % 4 trilinearities
        // ind2=4*(i-1);
        int ind2 = 4 * i;
        
        // S2=[0 -1; -1 0; x2(2) x2(1)];
        S2(2, 0) = x2(1);
        S2(2, 1) = x2(0);

        // S3=[0 -1; -1 0; x3(2) x3(1)];
        S3(2, 0) = x3(1);
        S3(2, 1) = x3(0);

        // f(ind2+1:ind2+4)=reshape( S2.'*(x1(1)*T1+x1(2)*T2+T3)*S3,4,1);
        f(Eigen::seq(ind2 + 0, ind2 + 3), 0) = (S2.transpose() * (x1(0) * T1 + x1(1) * T2 + T3) * S3).reshaped(4, 1);

        // % Jacobians for the trilinearities
        // Ap(ind2+1:ind2+4,:)=kron(S3,S2).'*kron([x1;1].',eye(9));
        Ap(Eigen::seq(ind2 + 0, ind2 + 3), Eigen::all) = Eigen::kroneckerProduct(S3, S2).transpose() * Eigen::kroneckerProduct(x1.colwise().homogeneous().transpose(), I9);

        // B(ind2+1:ind2+4,ind+1)=reshape(S2.'*T1*S3,4,1);
        B(Eigen::seq(ind2 + 0, ind2 + 3), ind + 0) = (S2.transpose() * T1 * S3).reshaped(4, 1);
        
        // B(ind2+1:ind2+4,ind+2)=reshape(S2.'*T2*S3,4,1);
        B(Eigen::seq(ind2 + 0, ind2 + 3), ind + 1) = (S2.transpose() * T2 * S3).reshaped(4, 1);
        
        // B(ind2+1:ind2+4,ind+3:ind+4)=kron(S3.'*J3.'*[x1;1],[0,1;1,0]);
        tmm = S3.transpose() * J3.transpose() * x1.colwise().homogeneous();
        B(Eigen::seq(ind2 + 0, ind2 + 3), Eigen::seq(ind + 2, ind + 3)) = Eigen::kroneckerProduct(tmm, tmp_22_swp);
        
        // B(ind2+1:ind2+4,ind+5:ind+6)=kron([0,1;1,0],S2.'*K3*[x1;1]);
        B(Eigen::seq(ind2 + 0, ind2 + 3), Eigen::seq(ind + 4, ind + 5)) = Eigen::kroneckerProduct(tmp_22_swp, S2.transpose() * K3 * x1.colwise().homogeneous());
    
    // end
    }

    // % jacobian for parametrization t=F(p) w.r.t. p evaluated in p
    // D=zeros(27,20);
    Eigen::MatrixXf D = Eigen::MatrixXf::Zero(27, 20);

    // D(:,1:9)=kron(eye(3),kron(eye(3),e21));
    D(Eigen::all, Eigen::seq(0, 8)) = Eigen::kroneckerProduct(I3, Eigen::kroneckerProduct(I3, e21));
    
    // aux=zeros(3,2); 
    // aux(Ind2,:)=eye(2);
    Eigen::Matrix<float, 3, 2> aux = Eigen::Matrix<float, 3, 2>::Zero();
    aux(Ind2, Eigen::all) = I2;

    // D(:,10:11)=[kron(S(:,1),aux);
    //             kron(S(:,2),aux);
    //             kron(S(:,3),aux)];
    Eigen::MatrixXf tmp_D_9_10(27, 2);
    tmp_D_9_10 << Eigen::kroneckerProduct(S.col(0), aux),
                  Eigen::kroneckerProduct(S.col(1), aux),
                  Eigen::kroneckerProduct(S.col(2), aux);
    D(Eigen::all, Eigen::seq(9, 10)) = tmp_D_9_10;

    // D(:,12:14)=kron(eye(3),kron(e31,aux(:,1)));
    D(Eigen::all, Eigen::seq(11, 13)) = Eigen::kroneckerProduct(I3, Eigen::kroneckerProduct(e31, aux.col(0)));
    
    // D(:,15:17)=kron(eye(3),kron(e31,aux(:,2)));
    D(Eigen::all, Eigen::seq(14, 16)) = Eigen::kroneckerProduct(I3, Eigen::kroneckerProduct(e31, aux.col(1)));

    // D(:,18:20)=[kron(eye(3),mn(1,:).');
    //             kron(eye(3),mn(2,:).');
    //             kron(eye(3),mn(3,:).')];
    Eigen::MatrixXf tmp_D_17_19(27, 3);
    tmp_D_17_19 << Eigen::kroneckerProduct(I3, mn.row(0).transpose()),
                   Eigen::kroneckerProduct(I3, mn.row(1).transpose()),
                   Eigen::kroneckerProduct(I3, mn.row(2).transpose());
    D(Eigen::all, Eigen::seq(17, 19)) = tmp_D_17_19;

    // % jacobian of f w.r.t. the minimal parameterization
    // A=Ap*D;
    Eigen::MatrixXf A = Ap * D; // Ap(4*N, 27) D(27,20) -> A(4*N, 20)

    // D=zeros(2,0);
    D = Eigen::MatrixXf::Zero(2, 0);

    // f(4*N, 1)
    // g(2, 1)
    // A(4*N, 20)
    // B(4*N, 6*N)
    // C(2, 20)
    // D(2, 0)
    return { f, g, A, B, C, D };
}

gh_result 
Gauss_Helmert
(
    gh_function f,
    Eigen::Ref<const Eigen::MatrixXf> const& x0, // (6*N)x1
    Eigen::Ref<const Eigen::MatrixXf> const& t0, // 20x1
    Eigen::Ref<const Eigen::MatrixXf> const& y0, // ?x? // 0x1
    Eigen::Ref<const Eigen::MatrixXf> const& x,  // (6xN)x1
    Eigen::Ref<const Eigen::MatrixXf> const& P,  // (6*N)x(6*N)
    void* user
)
{
    float rs = 1e-12f;

    // it_max=400;
    int it_max = 400;

    // tol=1e-6;
    float tol = 1e-9f;

    // xi=x0; 
    // yi=y0; 
    // ti=t0;
    Eigen::MatrixXf xi = x0;
    Eigen::MatrixXf yi = y0;
    Eigen::MatrixXf ti = t0;

    // u=size(t0,1);
    int u = t0.rows();

    // s=size(y0,1);
    int s = y0.rows();

    // v0=x0-x;
    Eigen::MatrixXf v0 = x0 - x;

    // objFunc=sum(v0.'*P*v0);
    float objFunc = (v0.transpose() * P * v0).sum();
    
    // factor=1;
    float factor = 1.0f;
    
    Eigen::MatrixXf Pi = P.inverse();
    Eigen::MatrixXf z_us = Eigen::MatrixXf::Zero(u, s);
    Eigen::MatrixXf z_sus = Eigen::MatrixXf::Zero(s, u+s);
    Eigen::MatrixXf z_s = Eigen::MatrixXf::Zero(s, 1);

    Eigen::MatrixXf sIW;
    Eigen::MatrixXf sIM;
    Eigen::MatrixXf z_c2c2;
    Eigen::MatrixXf W;
    Eigen::MatrixXf w;
    Eigen::MatrixXf M;
    Eigen::MatrixXf M1;
    Eigen::MatrixXf M2;
    Eigen::MatrixXf M3;
    Eigen::MatrixXf b;
    Eigen::MatrixXf aux;
    Eigen::MatrixXf dt;
    Eigen::MatrixXf dy;
    Eigen::MatrixXf v;

    int c2_p = -1;
    int cw_p = -1;

    gh_evaluation ghe;

    // for it=1:it_max
    int it;
    for (it = 0; it < it_max; ++it)
    {
        // [f,g,A,B,C,D]=func(xi,ti,yi);
        ghe = f(xi, ti, yi, user);

        // c2=size(C,1);
        int c2 = ghe.C.rows();
        int m_size = u + s + c2;

        // W=B*pinv(P)*B.';
        W = ghe.B * Pi * ghe.B.transpose();
        int cw = W.rows();

        // if any(isnan(W(:))) || any(isinf(W(:)))
        if (!std::isfinite(W.sum()))
        {
            //std::cout << "STOP inf/nan W" << std::endl;
            // break;
            break;
        // end
        }

        if (c2 != c2_p)
        {
            c2_p = c2;
            sIM = rs * Eigen::MatrixXf::Identity(m_size, m_size);
            z_c2c2 = Eigen::MatrixXf::Zero(c2, c2);
            M1 = Eigen::MatrixXf(u, m_size);
            M2 = Eigen::MatrixXf(s, m_size);
            M3 = Eigen::MatrixXf(c2, m_size);
            M = Eigen::MatrixXf(m_size, m_size);
            b = Eigen::MatrixXf(m_size, 1);
        }

        if (cw != cw_p)
        {
            cw_p = cw;
            sIW = rs * Eigen::MatrixXf::Identity(cw, cw);
        }
        
        // W=pinv(W+(1e-12)*eye(size(W,1)));
        // W=W+(1e-12)*eye(size(W,1));
        W = (W + sIW).completeOrthogonalDecomposition().pseudoInverse();
        W = W + sIW;        

        // w=-f-B*(x-xi);
        w = -(ghe.f + ghe.B * (x - xi));

        // M=[A.'*W*A, zeros(u,s), C.';...
        M1 << (ghe.A.transpose() * W * ghe.A), z_us, ghe.C.transpose();
        //           zeros(s,u+s), D.';...
        M2 << z_sus, ghe.D.transpose();
        //         C, D, zeros(c2,c2)];
        M3 << ghe.C, ghe.D, z_c2c2;

        M << M1, M2, M3;

        // b=[A.'*W*w; zeros(s,1); -g];
        b << (ghe.A.transpose() * W * w), z_s, (-ghe.g);

        // if any(isnan(M(:))) || any(isinf(M(:)))
        if (!std::isfinite(M.sum()))
        {
            //std::cout << "STOP inf/nan M" << std::endl;
            // break;
            break;
        // end
        }

        // aux=pinv(M+(1e-12)*eye(size(M,1)))*b;
        aux = (M + sIM).completeOrthogonalDecomposition().solve(b);

        // dt=aux(1:u,:); 
        // dy=aux(u+1:u+s,:);
        dt = aux(Eigen::seq(0, u - 1), Eigen::all);
        dy = aux(Eigen::seq(u, u + s - 1), Eigen::all);

        // v=-inv(P)*B.'*(W*(A*dt-w));
        v = -(Pi * ghe.B.transpose() * (W * (ghe.A * dt - w)));

        // if norm(dt)< tol && norm(dy) < tol && norm(xi-x-v) <tol
        if ((dt.norm() < tol) && (dy.norm() < tol) && ((xi - x - v).norm() < tol))
        {
            //std::cout << "STOP tol" << std::endl;
            // break;
            break;
        // end
        }
        
        // if sum(v.'*P*v) > objFunc*factor
        float of = (v.transpose() * P * v).sum();
        //std::cout << "cost: " << of << std::endl;
        if (of > objFunc * factor)
        {
            //std::cout << "STOP objFunc" << std::endl;
            // break;
            break;
        }
        else
        // else
        {
            // objFunc=sum(v.'*P*v);
            objFunc = of;
        // end
        }

        // xi=x+v; 
        // ti=ti+dt;
        // yi=yi+dy;
        xi = x + v;
        ti = ti + dt;
        yi = yi + dy;

    // end
    }

    // iter=it;
    // x_opt = xi; y_opt = yi; t_opt = ti;
    return { xi, ti, yi, it };
}

// OK
bool
trifocal_R_t_Ressl
(
    float const* p2d_1,
    float const* p2d_2,
    float const* p2d_3,
    float const* p2d_s,
    float const* p3d_s,
    int N,
    bool use_prior,
    float* r1,
    float* t1,
    float* r2,
    float* t2
)
{
    Eigen::MatrixXf p2d_1_w(2, N);
    Eigen::MatrixXf p2d_2_w(2, N);
    Eigen::MatrixXf p2d_3_w(2, N);
    Eigen::MatrixXf p2d_s_w(2, N);
    Eigen::MatrixXf p3d_s_w(3, N);

    memcpy(p2d_s_w.data(), p2d_s, 2 * N * sizeof(float));
    memcpy(p3d_s_w.data(), p3d_s, 3 * N * sizeof(float));
    memcpy(p2d_1_w.data(), p2d_1, 2 * N * sizeof(float));
    memcpy(p2d_2_w.data(), p2d_2, 2 * N * sizeof(float));
    memcpy(p2d_3_w.data(), p2d_3, 2 * N * sizeof(float));

    Eigen::MatrixXf p2d(6, N);
    p2d << p2d_1_w, p2d_2_w, p2d_3_w;

    Eigen::MatrixXf At = build_A(p2d_1_w, p2d_2_w, p2d_3_w);
    result_linear_TFT ltr = linear_TFT(At);

    Eigen::Matrix<float, 3, 4> P1 = Eigen::Matrix<float, 3, 4>::Identity(); // TODO: constant

    // e21=P2(:,4);
    Eigen::Matrix<float, 3, 1> e21 = ltr.P2.col(3);

    // [~,Ind]=max(abs(e21));
    // e21=e21/e21(Ind);
    Eigen::Index Ind;
    e21.cwiseAbs().maxCoeff(&Ind);
    //std::cout << e21 << std::endl;
   // std::cout << Ind << std::endl;
    e21 = e21 / e21(Ind);

    // e31=P3(:,4);           
    // e31=e31/norm(e31);
    Eigen::Matrix<float, 3, 1> e31 = ltr.P3.col(3);
    e31.normalize();

    // S=[T(Ind,:,1).',T(Ind,:,2).',T(Ind,:,3).'];
    Eigen::Matrix<float, 3, 3> T1 = ltr.TFT(Eigen::seqN( 0, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> T2 = ltr.TFT(Eigen::seqN( 9, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> T3 = ltr.TFT(Eigen::seqN(18, 9)).reshaped(3, 3);

    Eigen::Matrix<float, 3, 3> S;

    S.col(0) = T1.row(Ind).transpose();
    S.col(1) = T2.row(Ind).transpose();
    S.col(2) = T3.row(Ind).transpose();

    //aux=norm(S(:));
    //S=S./aux;
    float aux = S.norm();
    S = S / aux;

    // T=T/aux;
    //ltr.TFT = ltr.TFT / aux;
    T1 = T1 / aux;
    T2 = T2 / aux;
    T3 = T3 / aux;

    // Ind2=1:3; 
    // Ind2=Ind2(Ind2~=Ind);
    Eigen::Index Ind2[2];
    switch (Ind)
    {
    case 0: Ind2[0] = 1; Ind2[1] = 2; break;
    case 1: Ind2[0] = 0; Ind2[1] = 2; break;
    case 2: Ind2[0] = 0; Ind2[1] = 1; break;
    default:  std::cout << "BUGGGGGGG" << std::endl;                        break; // TODO: BUG
    }
    
    // mn=[e31.'*(T(:,:,1).'-S(:,1)*e21.');...
    //     e31.'*(T(:,:,2).'-S(:,2)*e21.');...
    //     e31.'*(T(:,:,3).'-S(:,3)*e21.')];
    Eigen::Matrix<float, 3, 3> mn_33;
    
    mn_33.row(0) = e31.transpose() * (T1.transpose() - S.col(0) * e21.transpose());
    mn_33.row(1) = e31.transpose() * (T2.transpose() - S.col(1) * e21.transpose());
    mn_33.row(2) = e31.transpose() * (T3.transpose() - S.col(2) * e21.transpose());

    // mn=mn(:,Ind2);
    Eigen::Matrix<float, 3, 2> mn_32;
    mn_32.col(0) = mn_33.col(Ind2[0]);
    mn_32.col(1) = mn_33.col(Ind2[1]);

    //std::cout << "mn_33" << std::endl;
    //std::cout << mn_33 << std::endl;
    
    // points3D=triangulation3D({P1,P2,P3},[x1;x2;x3]);
    Eigen::Matrix<float, 3, 4 * 3> PX;
    PX << P1, ltr.P2, ltr.P3;
    Eigen::MatrixXf p3dh = triangulate(PX, p2d);

    // p1_est=P1*points3D; p1_est=p1_est(1:2,:)./repmat(p1_est(3,:),2,1);
    Eigen::MatrixXf p1_est = (    P1 * p3dh).colwise().hnormalized();

    // p2_est=P2*points3D; p2_est=p2_est(1:2,:)./repmat(p2_est(3,:),2,1);
    Eigen::MatrixXf p2_est = (ltr.P2 * p3dh).colwise().hnormalized();

    // p3_est=P3*points3D; p3_est=p3_est(1:2,:)./repmat(p3_est(3,:),2,1);
    Eigen::MatrixXf p3_est = (ltr.P3 * p3dh).colwise().hnormalized();



    // N=size(x1,2);
    // p=[S(:);e21(Ind2);mn(:);e31];
    Eigen::MatrixXf p(9 + 2 + 6 + 3, 1);

    p << S.reshaped(9, 1),
        e21(Ind2[0]),
        e21(Ind2[1]),
         mn_32.reshaped(6, 1),
         e31;

    // x=reshape([x1(1:2,:);x2(1:2,:);x3(1:2,:)],6*N,1);
    Eigen::MatrixXf x(6 * N, 1);
    x = p2d.reshaped(6 * N, 1);

    Eigen::MatrixXf x_est(6, N);
    x_est << p1_est, p2_est, p3_est;
    x_est = x_est.reshaped(6 * N, 1);

    Eigen::MatrixXf y(0, 1);
    Eigen::MatrixXf P = Eigen::MatrixXf::Identity(6 * N, 6 * N);
    gh_result ghr = Gauss_Helmert(gh_ressl, x_est, p, y, x, P, &Ind);
    //std::cout << "GH ITER: " << ghr.iter << std::endl;
    ghr.t_opt = p;

    S = ghr.t_opt(Eigen::seqN(0, 9), 0).reshaped(3, 3);

    e21 = Eigen::Matrix<float, 3, 1>::Ones();
    e21(Ind2[0]) = ghr.t_opt(9, 0);
    e21(Ind2[1]) = ghr.t_opt(10, 0);

    mn_33 = Eigen::Matrix<float, 3, 3>::Zero();
    mn_33.col(Ind2[0]) = ghr.t_opt(Eigen::seqN(11, 3), 0);
    mn_33.col(Ind2[1]) = ghr.t_opt(Eigen::seqN(14, 3), 0);
    e31 = ghr.t_opt(Eigen::seqN(17, 3), 0);

    T1 = ((S.col(0) * e21.transpose()) + (e31 * mn_33.row(0))).transpose();
    T2 = ((S.col(1) * e21.transpose()) + (e31 * mn_33.row(1))).transpose();
    T3 = ((S.col(2) * e21.transpose()) + (e31 * mn_33.row(2))).transpose();

    ltr.TFT << T1.reshaped(9, 1), T2.reshaped(9, 1), T3.reshaped(9, 1);

    result_R_t_from_TFT rpt = R_t_from_TFT(ltr.TFT, p2d_1_w, p2d_2_w, p2d_3_w, p3d_s_w, use_prior);

    float world_scale = compute_scale(p2d_s_w, p3d_s_w, rpt.v2.P);

    Eigen::Matrix<float, 3, 3> R2 = rpt.v2.P(Eigen::all, Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 3> R3 = rpt.v3.P(Eigen::all, Eigen::seqN(0, 3));

    Eigen::AngleAxis<float> R01(R2);
    Eigen::AngleAxis<float> R02(R3);

    Eigen::Matrix<float, 3, 1> r01 = R01.angle() * R01.axis();
    Eigen::Matrix<float, 3, 1> r02 = R02.angle() * R02.axis();

    Eigen::Matrix<float, 3, 1> t01 = world_scale * rpt.v2.P.col(3);
    Eigen::Matrix<float, 3, 1> t02 = world_scale * rpt.v3.P.col(3);

    memcpy(r1, r01.data(), 3 * sizeof(float));
    memcpy(r2, r02.data(), 3 * sizeof(float));
    memcpy(t1, t01.data(), 3 * sizeof(float));
    memcpy(t2, t02.data(), 3 * sizeof(float));

    return std::isfinite(r1[0] + r1[1] + r1[2] + t1[0] + t1[1] + t1[2] + r2[0] + r2[1] + r2[2] + t2[0] + t2[1] + t2[2]);
}













// OK
bool trifocal_R_t(float const* p2d_1, float const* p2d_2, float const* p2d_3, float const* sp2d, float const* sp3d, float* tft, float* rt1, float* rt2, float* s1, float* s2)
{
    return trifocal_R_t_Ressl(p2d_1, p2d_2, p2d_3, sp2d, sp3d, 7, false, rt1 + 0, rt1 + 3, rt2 + 0, rt2 + 3);
}

struct trifocal_data_map
{
    float const* p2d_1;
    float const* p2d_2;
    float const* p2d_3;
    float const* p2d_s;
    float const* p3d_s;
    float* tft;
    float* rt1;
    float* rt2;
    float* s1;
    float* s2;
    int count;
};

struct trifocal_job_descriptor
{
    trifocal_data_map const* map;
    int const* rng;
    int id;
    int start;
    int end;
    int valid;
};

// OK
void trifocal_R_t_group(trifocal_job_descriptor& tjd)
{
    bool enable_world_scale = (tjd.map->p2d_s) && (tjd.map->p3d_s);

    float p2d_1[2 * 7];
    float p2d_2[2 * 7];
    float p2d_3[2 * 7];
    float p2d_s[2 * 7];
    float p3d_s[3 * 7];

    float const* base_p2d_s = enable_world_scale ? p2d_s : nullptr;
    float const* base_p3d_s = enable_world_scale ? p3d_s : nullptr;

    tjd.valid = 0;

    for (int i = tjd.start; i < tjd.end; ++i)
    {
        for (int p = 0; p < 7; ++p)
        {
            int q = tjd.rng[(7 * i) + p];
            
            p2d_1[(2 * p) + 0] = tjd.map->p2d_1[(2 * q) + 0];
            p2d_1[(2 * p) + 1] = tjd.map->p2d_1[(2 * q) + 1];
            p2d_2[(2 * p) + 0] = tjd.map->p2d_2[(2 * q) + 0];
            p2d_2[(2 * p) + 1] = tjd.map->p2d_2[(2 * q) + 1];
            p2d_3[(2 * p) + 0] = tjd.map->p2d_3[(2 * q) + 0];
            p2d_3[(2 * p) + 1] = tjd.map->p2d_3[(2 * q) + 1];

            if (!enable_world_scale) { continue; }

            p2d_s[(2 * p) + 0] = tjd.map->p2d_s[(2 * q) + 0];
            p2d_s[(2 * p) + 1] = tjd.map->p2d_s[(2 * q) + 1];
            p3d_s[(3 * p) + 0] = tjd.map->p3d_s[(3 * q) + 0];
            p3d_s[(3 * p) + 1] = tjd.map->p3d_s[(3 * q) + 1];
            p3d_s[(3 * p) + 2] = tjd.map->p3d_s[(3 * q) + 2];
        }

        int offset = tjd.start + tjd.valid;

        float* base_tft = tjd.map->tft ? tjd.map->tft + (27 * offset) : nullptr;
        float* base_s1  = tjd.map->s1  ? tjd.map->s1  + ( 1 * offset) : nullptr;
        float* base_s2  = tjd.map->s2  ? tjd.map->s2  + ( 1 * offset) : nullptr;
        float* base_rt1 = tjd.map->rt1 + (6 * offset);
        float* base_rt2 = tjd.map->rt2 + (6 * offset);
        
        bool valid = trifocal_R_t(p2d_1, p2d_2, p2d_3, base_p2d_s, base_p3d_s, base_tft, base_rt1, base_rt2, base_s1, base_s2);

        tjd.valid += valid; // TODO: Notify if not valid
    }
}

// OK
int trifocal_R_t_batch(int jobs, int workers, float const* p2d_1, float const* p2d_2, float const* p2d_3, float const* p2d_s, float const* p3d_s, int count, float* tft, float* rt1, float* rt2, float* s1, float* s2)
{
    trifocal_data_map map{ p2d_1, p2d_2, p2d_3, p2d_s, p3d_s, tft, rt1, rt2, s1, s2, count };

    int batch = jobs / workers;
    int spill = jobs % workers;
    int start = 0;
    int valid = 0;

    std::unique_ptr<int[]> rng = std::make_unique<int[]>(7 * jobs);

    std::vector<trifocal_job_descriptor> registry;
    std::vector<std::thread> threads;

    for (int i = 0; i < jobs; ++i)
    {
        for (int p = 0; p < 7; ++p)
        {
            rng[(7 * i) + p] = ((float)rand() / (float)RAND_MAX) * count;
        }
    }

    for (int i = 0; i < workers; ++i)
    {
        int end = start + batch;
        if (spill > 0)
        {
            end++;
            spill--;
        }
        if (start >= end) { break; }
        registry.push_back({ &map, rng.get(), i, start, end, 0 });
        start = end;
    }

    for (auto& tjd : registry)
    {
        threads.push_back(std::thread(trifocal_R_t_group, std::ref(tjd)));
    }

    for (auto& wtp : threads)
    {
        wtp.join();
    }

    for (auto& tjd : registry)
    {
        if (tjd.valid <= 0) { continue; }

        memmove(rt1 + (6 * valid), rt1 + (6 * tjd.start), (6 * tjd.valid) * sizeof(float));
        memmove(rt2 + (6 * valid), rt2 + (6 * tjd.start), (6 * tjd.valid) * sizeof(float));

        if (tft) { memmove(tft + (27 * valid), tft + (27 * tjd.start), (27 * tjd.valid) * sizeof(float)); }
        if (s1)  { memmove(s1  + ( 1 * valid), s1  + ( 1 * tjd.start), ( 1 * tjd.valid) * sizeof(float)); }
        if (s2)  { memmove(s2  + ( 1 * valid), s2  + ( 1 * tjd.start), ( 1 * tjd.valid) * sizeof(float)); }

        valid += tjd.valid;
    }

    return valid;
}
