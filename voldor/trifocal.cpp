
#define EIGEN_NO_AUTOMATIC_RESIZING

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h>

//-----------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------

// OK
void build_A(float const* p2d_1, float const* p2d_2, float const* p2d_3, Eigen::Ref<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> result)
{
    float* A = result.data();

    for (int i = 0; i < 7; ++i)
    {
        float x1 = p2d_1[(i * 2) + 0];
        float y1 = p2d_1[(i * 2) + 1];
        float x2 = p2d_2[(i * 2) + 0];
        float y2 = p2d_2[(i * 2) + 1];
        float x3 = p2d_3[(i * 2) + 0];
        float y3 = p2d_3[(i * 2) + 1];

        A[(((4 * i) + 0) * 27) +  0] = x1;
        A[(((4 * i) + 0) * 27) +  1] = 0;
        A[(((4 * i) + 0) * 27) +  2] = -x1 * x2;
        A[(((4 * i) + 0) * 27) +  3] = 0;
        A[(((4 * i) + 0) * 27) +  4] = 0;
        A[(((4 * i) + 0) * 27) +  5] = 0;
        A[(((4 * i) + 0) * 27) +  6] = -x1 * x3;
        A[(((4 * i) + 0) * 27) +  7] = 0;
        A[(((4 * i) + 0) * 27) +  8] = x1 * x2 * x3;
        A[(((4 * i) + 0) * 27) +  9] = y1;
        A[(((4 * i) + 0) * 27) + 10] = 0;
        A[(((4 * i) + 0) * 27) + 11] = -x2 * y1;
        A[(((4 * i) + 0) * 27) + 12] = 0;
        A[(((4 * i) + 0) * 27) + 13] = 0;
        A[(((4 * i) + 0) * 27) + 14] = 0;
        A[(((4 * i) + 0) * 27) + 15] = -x3 * y1;
        A[(((4 * i) + 0) * 27) + 16] = 0;
        A[(((4 * i) + 0) * 27) + 17] = x2 * x3 * y1;
        A[(((4 * i) + 0) * 27) + 18] = 1;
        A[(((4 * i) + 0) * 27) + 19] = 0;
        A[(((4 * i) + 0) * 27) + 20] = -x2;
        A[(((4 * i) + 0) * 27) + 21] = 0;
        A[(((4 * i) + 0) * 27) + 22] = 0;
        A[(((4 * i) + 0) * 27) + 23] = 0;
        A[(((4 * i) + 0) * 27) + 24] = -x3;
        A[(((4 * i) + 0) * 27) + 25] = 0;
        A[(((4 * i) + 0) * 27) + 26] = x2 * x3;

        A[(((4 * i) + 1) * 27) +  0] = 0;
        A[(((4 * i) + 1) * 27) +  1] = x1;
        A[(((4 * i) + 1) * 27) +  2] = -x1 * y2;
        A[(((4 * i) + 1) * 27) +  3] = 0;
        A[(((4 * i) + 1) * 27) +  4] = 0;
        A[(((4 * i) + 1) * 27) +  5] = 0;
        A[(((4 * i) + 1) * 27) +  6] = 0;
        A[(((4 * i) + 1) * 27) +  7] = -x1 * x3;
        A[(((4 * i) + 1) * 27) +  8] = x1 * x3 * y2;
        A[(((4 * i) + 1) * 27) +  9] = 0;
        A[(((4 * i) + 1) * 27) + 10] = y1;
        A[(((4 * i) + 1) * 27) + 11] = -y1 * y2;
        A[(((4 * i) + 1) * 27) + 12] = 0;
        A[(((4 * i) + 1) * 27) + 13] = 0;
        A[(((4 * i) + 1) * 27) + 14] = 0;
        A[(((4 * i) + 1) * 27) + 15] = 0;
        A[(((4 * i) + 1) * 27) + 16] = -x3 * y1;
        A[(((4 * i) + 1) * 27) + 17] = x3 * y1 * y2;
        A[(((4 * i) + 1) * 27) + 18] = 0;
        A[(((4 * i) + 1) * 27) + 19] = 1;
        A[(((4 * i) + 1) * 27) + 20] = -y2;
        A[(((4 * i) + 1) * 27) + 21] = 0;
        A[(((4 * i) + 1) * 27) + 22] = 0;
        A[(((4 * i) + 1) * 27) + 23] = 0;
        A[(((4 * i) + 1) * 27) + 24] = 0;
        A[(((4 * i) + 1) * 27) + 25] = -x3;
        A[(((4 * i) + 1) * 27) + 26] = x3 * y2;

        A[(((4 * i) + 2) * 27) +  0] = 0;
        A[(((4 * i) + 2) * 27) +  1] = 0;
        A[(((4 * i) + 2) * 27) +  2] = 0;
        A[(((4 * i) + 2) * 27) +  3] = x1;
        A[(((4 * i) + 2) * 27) +  4] = 0;
        A[(((4 * i) + 2) * 27) +  5] = -x1 * x2;
        A[(((4 * i) + 2) * 27) +  6] = -x1 * y3;
        A[(((4 * i) + 2) * 27) +  7] = 0;
        A[(((4 * i) + 2) * 27) +  8] = x1 * x2 * y3;
        A[(((4 * i) + 2) * 27) +  9] = 0;
        A[(((4 * i) + 2) * 27) + 10] = 0;
        A[(((4 * i) + 2) * 27) + 11] = 0;
        A[(((4 * i) + 2) * 27) + 12] = y1;
        A[(((4 * i) + 2) * 27) + 13] = 0;
        A[(((4 * i) + 2) * 27) + 14] = -x2 * y1;
        A[(((4 * i) + 2) * 27) + 15] = -y1 * y3;
        A[(((4 * i) + 2) * 27) + 16] = 0;
        A[(((4 * i) + 2) * 27) + 17] = x2 * y1 * y3;
        A[(((4 * i) + 2) * 27) + 18] = 0;
        A[(((4 * i) + 2) * 27) + 19] = 0;
        A[(((4 * i) + 2) * 27) + 20] = 0;
        A[(((4 * i) + 2) * 27) + 21] = 1;
        A[(((4 * i) + 2) * 27) + 22] = 0;
        A[(((4 * i) + 2) * 27) + 23] = -x2;
        A[(((4 * i) + 2) * 27) + 24] = -y3;
        A[(((4 * i) + 2) * 27) + 25] = 0;
        A[(((4 * i) + 2) * 27) + 26] = x2 * y3;

        A[(((4 * i) + 3) * 27) +  0] = 0;
        A[(((4 * i) + 3) * 27) +  1] = 0;
        A[(((4 * i) + 3) * 27) +  2] = 0;
        A[(((4 * i) + 3) * 27) +  3] = 0;
        A[(((4 * i) + 3) * 27) +  4] = x1;
        A[(((4 * i) + 3) * 27) +  5] = -x1 * y2;
        A[(((4 * i) + 3) * 27) +  6] = 0;
        A[(((4 * i) + 3) * 27) +  7] = -x1 * y3;
        A[(((4 * i) + 3) * 27) +  8] = x1 * y2 * y3;
        A[(((4 * i) + 3) * 27) +  9] = 0;
        A[(((4 * i) + 3) * 27) + 10] = 0;
        A[(((4 * i) + 3) * 27) + 11] = 0;
        A[(((4 * i) + 3) * 27) + 12] = 0;
        A[(((4 * i) + 3) * 27) + 13] = y1;
        A[(((4 * i) + 3) * 27) + 14] = -y1 * y2;
        A[(((4 * i) + 3) * 27) + 15] = 0;
        A[(((4 * i) + 3) * 27) + 16] = -y1 * y3;
        A[(((4 * i) + 3) * 27) + 17] = y1 * y2 * y3;
        A[(((4 * i) + 3) * 27) + 18] = 0;
        A[(((4 * i) + 3) * 27) + 19] = 0;
        A[(((4 * i) + 3) * 27) + 20] = 0;
        A[(((4 * i) + 3) * 27) + 21] = 0;
        A[(((4 * i) + 3) * 27) + 22] = 1;
        A[(((4 * i) + 3) * 27) + 23] = -y2;
        A[(((4 * i) + 3) * 27) + 24] = 0;
        A[(((4 * i) + 3) * 27) + 25] = -y3;
        A[(((4 * i) + 3) * 27) + 26] = y2 * y3;
    }
}

// OK
void epipoles_from_TFT(Eigen::Ref<const Eigen::Matrix<float, 27, 1>> const& TFT, Eigen::Ref<Eigen::Matrix<float, 3, 2>> e)
{
    Eigen::Matrix<float, 3, 3> t1 = TFT(Eigen::seqN( 0, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> t2 = TFT(Eigen::seqN( 9, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> t3 = TFT(Eigen::seqN(18, 9)).reshaped(3, 3);

    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd_t1 = t1.jacobiSvd(Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd_t2 = t2.jacobiSvd(Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd_t3 = t3.jacobiSvd(Eigen::ComputeFullV | Eigen::ComputeFullU);

    Eigen::Matrix<float, 3, 3> vx;
    Eigen::Matrix<float, 3, 3> ux;

    vx << ( svd_t1.matrixV().col(2)),
          ( svd_t2.matrixV().col(2)),
          ( svd_t3.matrixV().col(2));

    ux << (-svd_t1.matrixU().col(2)),
          (-svd_t2.matrixU().col(2)),
          (-svd_t3.matrixU().col(2));

    e << (-ux.jacobiSvd(Eigen::ComputeFullU).matrixU().col(2)),
         (-vx.jacobiSvd(Eigen::ComputeFullU).matrixU().col(2));
}

// OK
void linear_TFT(Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> const& A, Eigen::Ref<Eigen::Matrix<float, 27, 1>> result, float threshold = 0)
{
    Eigen::Matrix<float, 27, 1> t = -(A.bdcSvd(Eigen::ComputeThinU).matrixU().col(26));

    Eigen::Matrix<float, 3, 2> e;

    epipoles_from_TFT(t, e); // OK

    Eigen::Matrix<float, 3, 3> I3 = Eigen::Matrix<float, 3, 3>::Identity();
    Eigen::Matrix<float, 9, 9> I9 = Eigen::Matrix<float, 9, 9>::Identity();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> E(27, 18);

    E << Eigen::kroneckerProduct(I3, Eigen::kroneckerProduct(e.col(1), I3)),
         Eigen::kroneckerProduct(I9,                        -e.col(0));

    Eigen::BDCSVD<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> svd_E = E.bdcSvd(Eigen::ComputeThinU);
    if (threshold > 0) { svd_E.setThreshold(threshold); }
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Up = svd_E.matrixU()(Eigen::all, Eigen::seqN(0, svd_E.rank()));

    result = Up * ((A.transpose() * Up).bdcSvd(Eigen::ComputeThinV).matrixV()(Eigen::all, Eigen::last));
}

// OK
void cross_matrix(Eigen::Ref<const Eigen::Matrix<float, 3, 1>> const& v, Eigen::Ref<Eigen::Matrix<float, 3, 3>> M)
{
    M <<     0,   (-v(2)), ( v(1)),
         ( v(2)),     0,   (-v(0)),
         (-v(1)), ( v(0)),     0;
}

// OK
void triangulate(Eigen::Ref<const Eigen::Matrix<float, 3, 4 * 2>> const& cameras, float const* p2d_1, float const* p2d_2, Eigen::Ref<Eigen::Matrix<float, 4, 7>> p3h)
{
    Eigen::Matrix<float, 2 * 2, 4> ls_matrix;
    Eigen::Matrix<float, 2, 3> L
    {
        {0.0f, -1.0f, 0.0f},
        {1.0f,  0.0f, 0.0f},
    };

    for (int n = 0; n < 7; ++n)
    {
        L(0, 2) =  p2d_1[(n * 2) + 1];
        L(1, 2) = -p2d_1[(n * 2) + 0];

        ls_matrix(Eigen::seqN(0 * 2, 2), Eigen::all) = L * cameras(Eigen::all, Eigen::seqN(0 * 4, 4));

        L(0, 2) =  p2d_2[(n * 2) + 1];
        L(1, 2) = -p2d_2[(n * 2) + 0];

        ls_matrix(Eigen::seqN(1 * 2, 2), Eigen::all) = L * cameras(Eigen::all, Eigen::seqN(1 * 4, 4));

        p3h.col(n) = ls_matrix.bdcSvd(Eigen::ComputeFullV).matrixV().col(3);
    }
}

// OK
void R_t_from_E(Eigen::Ref<const Eigen::Matrix<float, 3, 3>> const& E, float const* p2d_1, float const* p2d_2, Eigen::Ref<Eigen::Matrix<float, 3, 4>> P, Eigen::Ref<Eigen::Matrix<float, 4, 7>> p3h)
{
    Eigen::Matrix<float, 3, 3> W
    {
        {0.0f, -1.0f, 0.0f},
        {1.0f,  0.0f, 0.0f},
        {0.0f,  0.0f, 1.0f},
    };

    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> E_svd = E.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<float, 3, 3> U  = E_svd.matrixU();
    Eigen::Matrix<float, 3, 3> Vt = E_svd.matrixV().transpose();

    Eigen::Matrix<float, 3, 3> R1 = U * W             * Vt;
    Eigen::Matrix<float, 3, 3> R2 = U * W.transpose() * Vt;

    if (R1.determinant() < 0) { R1 = -R1; }
    if (R2.determinant() < 0) { R2 = -R2; }

    Eigen::Matrix<float, 3, 1> pt = U.col(2);
    Eigen::Matrix<float, 3, 1> nt = -pt;

    Eigen::Matrix<float, 3, 4> P0 = Eigen::Matrix<float, 3, 4>::Identity();
    Eigen::Matrix<float, 3, 4> P1;
    Eigen::Matrix<float, 3, 4 * 2> PX;
    Eigen::Matrix<float, 4, 7> XYZW;

    int64_t max_count = -1;

    for (int i = 0; i < 4; ++i)
    {
        switch (i)
        {
        case 0: P1 << R1, pt; break;
        case 1: P1 << R1, nt; break;
        case 2: P1 << R2, pt; break;
        case 3: P1 << R2, nt; break;
        }

        PX << P0, P1;

        triangulate(PX, p2d_1, p2d_2, XYZW); // OK

        int64_t count = (XYZW.colwise().hnormalized().row(2).array() > 0).count() + ((P1 * XYZW).row(2).array() > 0).count();
        if (count < max_count) { continue; }
        max_count = count;

        P = P1;
        p3h = XYZW;
    }
}

// OK
float R_t_from_TFT(Eigen::Ref<const Eigen::Matrix<float, 27, 1>> const& TFT, float const* p2d_1, float const* p2d_2, float const* p2d_3, Eigen::Ref<Eigen::Matrix<float, 3, 4>> c1, Eigen::Ref<Eigen::Matrix<float, 3, 4>> c2)
{
    Eigen::Matrix<float, 3, 2> e;

    epipoles_from_TFT(TFT, e); // OK

    if (e(2, 0) < 0) { e.col(0) = -e.col(0); }
    if (e(2, 1) < 0) { e.col(1) = -e.col(1); }

    e.col(1) = -e.col(1);

    Eigen::Matrix<float, 3, 3> epi21_x;
    Eigen::Matrix<float, 3, 3> epi31_x;

    cross_matrix(e.col(0), epi21_x); // OK
    cross_matrix(e.col(1), epi31_x); // OK

    Eigen::Matrix<float, 3, 3> T1 = TFT(Eigen::seqN( 0, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> T2 = TFT(Eigen::seqN( 9, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> T3 = TFT(Eigen::seqN(18, 9)).reshaped(3, 3);

    Eigen::Matrix<float, 3, 3> D21;
    Eigen::Matrix<float, 3, 3> D31;

    D21 << (T1             * e.col(1)), (T2             * e.col(1)), (T3             * e.col(1));
    D31 << (T1.transpose() * e.col(0)), (T2.transpose() * e.col(0)), (T3.transpose() * e.col(0));

    Eigen::Matrix<float, 3, 3> E21 = epi21_x * D21;
    Eigen::Matrix<float, 3, 3> E31 = epi31_x * D31;

    Eigen::Matrix<float, 4, 7> p3DH(4, 7);

    R_t_from_E(E31, p2d_1, p2d_3, c2, p3DH); // OK
    R_t_from_E(E21, p2d_1, p2d_2, c1, p3DH); // OK

    Eigen::Matrix<float, 3, 7> X3 = c2(Eigen::all, Eigen::seqN(0, 3)) * p3DH.colwise().hnormalized();
    Eigen::Matrix<float, 3, 1> p3;
    Eigen::Matrix<float, 3, 1> p3_X3;
    Eigen::Matrix<float, 3, 1> p3_t3;

    float num = 0;
    float den = 0;

    for (int i = 0; i < 7; ++i)
    {
        p3(0) = p2d_3[(i * 2) + 0];
        p3(1) = p2d_3[(i * 2) + 1];
        p3(2) = 1.0f;

        p3_X3 = p3.cross(X3.col(i));
        p3_t3 = p3.cross(c2.col(3));

        num -= p3_X3.dot(p3_t3);
        den += p3_t3.dot(p3_t3);
    }

    float scale = num / den;

    c2.col(3) = scale * c2.col(3);

    return scale;
}

// OK
float compute_scale(float const* p2d, float const* p3d, Eigen::Ref<const Eigen::Matrix<float, 3, 4>> const& c1)
{
    Eigen::Matrix<float, 3, 3> Ri = c1(Eigen::all, Eigen::seqN(0, 3)).transpose();
    Eigen::Matrix<float, 3, 1> ti = -(Ri * c1.col(3));
    Eigen::Matrix<float, 3, 1> p3;
    Eigen::Matrix<float, 3, 1> p2;
    Eigen::Matrix<float, 3, 1> r2;

    float scales[7];
    float ws;
    int valid = 0;

    for (int i = 0; i < 7; ++i)
    {
        p3(0) = p3d[(3 * i) + 0];
        p3(1) = p3d[(3 * i) + 1];
        p3(2) = p3d[(3 * i) + 2];

        p2(0) = p2d[(2 * i) + 0];
        p2(1) = p2d[(2 * i) + 1];
        p2(2) = 1.0f;
        
        r2 = Ri * p2;
        ws = std::sqrt(r2.cross(p3).squaredNorm() / r2.cross(ti).squaredNorm());

        if (std::isfinite(ws)) { scales[valid++] = ws; }
    }

    if (valid <= 0) { return 0; }

    std::sort(scales, scales + valid);
    return scales[valid / 2];
}

// OK
void trifocal_R_t(float const* p2d_1, float const* p2d_2, float const* p2d_3, float const* sp2d, float const* sp3d, float* tft, float* r1, float* t1, float* r2, float* t2, float* s1, float* s2)
{
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(27, 4 * 7);
    Eigen::Matrix<float, 27, 1> TFT;
    Eigen::Matrix<float, 3, 4> P2;
    Eigen::Matrix<float, 3, 4> P3;

    build_A(p2d_1, p2d_2, p2d_3, A); // OK
    linear_TFT(A, TFT); // OK
    float local_scale = R_t_from_TFT(TFT, p2d_1, p2d_2, p2d_3, P2, P3); // OK
    float world_scale = compute_scale(sp2d, sp3d, P2);

    memcpy(tft, TFT.data(), 27 * sizeof(float));

    cv::Mat R01(3, 3, CV_32FC1, P2.data()); // needs transpose
    cv::Mat R02(3, 3, CV_32FC1, P3.data()); // needs transpose
    cv::Mat r01(3, 1, CV_32FC1, r1);
    cv::Mat r02(3, 1, CV_32FC1, r2);

    cv::Rodrigues(R01, r01);
    cv::Rodrigues(R02, r02);

    r01 *= -1;
    r02 *= -1;

    cv::Mat u01(3, 1, CV_32FC1, P2.data() + 9);
    cv::Mat u02(3, 1, CV_32FC1, P3.data() + 9);
    cv::Mat t01(3, 1, CV_32FC1, t1);
    cv::Mat t02(3, 1, CV_32FC1, t2);

    t01 = world_scale * u01;
    t02 = world_scale * u02;

    *s1 = world_scale;
    *s2 = local_scale;
}
