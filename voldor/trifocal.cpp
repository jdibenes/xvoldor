
#define EIGEN_NO_AUTOMATIC_RESIZING

#include <thread>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h>

//-----------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------

static void normalize_points(Eigen::Ref<const Eigen::Matrix<float, 2, 7>> const& p_i, Eigen::Ref<Eigen::Matrix<float, 2, 7>> p_o, Eigen::Ref<Eigen::Matrix<float, 3, 3>> N)
{
    Eigen::Matrix<float, 2, 1> p0 = p_i.rowwise().mean();

    float s = std::sqrt(2.0f) / (p_i.colwise() - p0).colwise().norm().mean();

    N = Eigen::Matrix<float, 3, 3>
    {
        {   s, 0.0f, -s * p0(0)},
        {0.0f,    s, -s * p0(1)},
        {0.0f, 0.0f,       1.0f},
    };
    
    p_o = N(Eigen::seqN(0, 2), Eigen::all) * p_i.colwise().homogeneous();
}


static void transform_TFT(Eigen::Ref<const Eigen::Matrix<float, 27, 1>> const& tft_i, Eigen::Ref<const Eigen::Matrix<float, 3, 3>> const& M1, Eigen::Ref<const Eigen::Matrix<float, 3, 3>> const& M2, Eigen::Ref<const Eigen::Matrix<float, 3, 3>> const& M3, Eigen::Ref<Eigen::Matrix<float, 27, 1>> tft_o, bool inverse)
{
    Eigen::Matrix<float, 3, 3> t1_i = tft_i(Eigen::seqN( 0, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> t2_i = tft_i(Eigen::seqN( 9, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> t3_i = tft_i(Eigen::seqN(18, 9)).reshaped(3, 3);

    Eigen::Matrix<float, 27, 1> t_o;
        
    if (!inverse)
    {
        Eigen::Matrix<float, 3, 3> M1i = M1.inverse();
        t_o << (M2 * (M1i(0, 0) * t1_i + M1i(1, 0) * t2_i + M1i(2, 0) * t3_i) * M3.transpose()).reshaped(9, 1),
               (M2 * (M1i(0, 1) * t1_i + M1i(1, 1) * t2_i + M1i(2, 1) * t3_i) * M3.transpose()).reshaped(9, 1),
               (M2 * (M1i(0, 2) * t1_i + M1i(1, 2) * t2_i + M1i(2, 2) * t3_i) * M3.transpose()).reshaped(9, 1);
    }
    else
    {
        Eigen::Matrix<float, 3, 3> M2i = M2.inverse();
        Eigen::Matrix<float, 3, 3> M3i = M3.inverse();
        t_o << (M2i * (M1(0, 0) * t1_i + M1(1, 0) * t2_i + M1(2, 0) * t3_i) * M3i.transpose()).reshaped(9, 1),
               (M2i * (M1(0, 1) * t1_i + M1(1, 1) * t2_i + M1(2, 1) * t3_i) * M3i.transpose()).reshaped(9, 1),
               (M2i * (M1(0, 2) * t1_i + M1(1, 2) * t2_i + M1(2, 2) * t3_i) * M3i.transpose()).reshaped(9, 1);
    }

    tft_o = t_o.normalized();
}

// OK
static void build_A(float const* p2d_1, float const* p2d_2, float const* p2d_3, Eigen::Ref<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> result)
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
static void epipoles_from_TFT(Eigen::Ref<const Eigen::Matrix<float, 27, 1>> const& TFT, Eigen::Ref<Eigen::Matrix<float, 3, 2>> e)
{
    Eigen::Matrix<float, 3, 3> t1 = TFT(Eigen::seqN( 0, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> t2 = TFT(Eigen::seqN( 9, 9)).reshaped(3, 3);
    Eigen::Matrix<float, 3, 3> t3 = TFT(Eigen::seqN(18, 9)).reshaped(3, 3);

    Eigen::BDCSVD<Eigen::Matrix<float, 3, 3>> svd_t1 = t1.bdcSvd(Eigen::ComputeFullV | Eigen::ComputeFullU); // Full = Thin
    Eigen::BDCSVD<Eigen::Matrix<float, 3, 3>> svd_t2 = t2.bdcSvd(Eigen::ComputeFullV | Eigen::ComputeFullU); // Full = Thin
    Eigen::BDCSVD<Eigen::Matrix<float, 3, 3>> svd_t3 = t3.bdcSvd(Eigen::ComputeFullV | Eigen::ComputeFullU); // Full = Thin

    Eigen::Matrix<float, 3, 3> vx;
    Eigen::Matrix<float, 3, 3> ux;

    vx << ( svd_t1.matrixV().col(2)),
          ( svd_t2.matrixV().col(2)),
          ( svd_t3.matrixV().col(2));

    ux << (-svd_t1.matrixU().col(2)),
          (-svd_t2.matrixU().col(2)),
          (-svd_t3.matrixU().col(2));

    e << (-ux.bdcSvd(Eigen::ComputeFullU).matrixU().col(2)), // Full = Thin
         (-vx.bdcSvd(Eigen::ComputeFullU).matrixU().col(2)); // Full = Thin
}

// OK
static void linear_TFT(Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> const& A, Eigen::Ref<Eigen::Matrix<float, 27, 1>> result, float threshold = 0)
{
    Eigen::Matrix<float, 27, 1> t = -(A.bdcSvd(Eigen::ComputeThinU).matrixU().col(26)); // Previously BDC SVD, jacobiSVD drift

    Eigen::Matrix<float, 3, 2> e;

    epipoles_from_TFT(t, e); // OK

    Eigen::Matrix<float, 3, 3> I3 = Eigen::Matrix<float, 3, 3>::Identity();
    Eigen::Matrix<float, 9, 9> I9 = Eigen::Matrix<float, 9, 9>::Identity();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> E(27, 18);

    E << Eigen::kroneckerProduct(I3, Eigen::kroneckerProduct(e.col(1), I3)),
         Eigen::kroneckerProduct(I9,                        -e.col(0));

    Eigen::BDCSVD<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> svd_E = E.bdcSvd(Eigen::ComputeThinU);  // BDC SVD gives NaN sometimes, Jacobi SVD doesn't
    if (threshold > 0) { svd_E.setThreshold(threshold); }
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Up = svd_E.matrixU()(Eigen::all, Eigen::seqN(0, svd_E.rank())); // Rank seems to be always 15

    result = Up * ((A.transpose() * Up).bdcSvd(Eigen::ComputeThinV).matrixV()(Eigen::all, Eigen::last)); // Previously BDC SVD
}

// OK
static void cross_matrix(Eigen::Ref<const Eigen::Matrix<float, 3, 1>> const& v, Eigen::Ref<Eigen::Matrix<float, 3, 3>> M)
{
    M <<     0,   (-v(2)), ( v(1)),
         ( v(2)),     0,   (-v(0)),
         (-v(1)), ( v(0)),     0;
}

// OK
static void triangulate(Eigen::Ref<const Eigen::Matrix<float, 3, 4>> const& P0, Eigen::Ref<const Eigen::Matrix<float, 3, 4>> const& P1, float const* p2d_1, float const* p2d_2, Eigen::Ref<Eigen::Matrix<float, 4, 7>> p3h)
{
    Eigen::Matrix<float, 2, 3> L0
    {
        {0.0f, -1.0f, 0.0f},
        {1.0f,  0.0f, 0.0f},
    };
    Eigen::Matrix<float, 2, 3> L1
    {
        {0.0f, -1.0f, 0.0f},
        {1.0f,  0.0f, 0.0f},
    };

    Eigen::Matrix<float, 2 * 2, 4> ls_matrix;
    Eigen::Matrix<float, 4, 1> XYZW;

    for (int n = 0; n < 7; ++n)
    {
        L0(0, 2) =  p2d_1[(n * 2) + 1];
        L0(1, 2) = -p2d_1[(n * 2) + 0];
        L1(0, 2) =  p2d_2[(n * 2) + 1];
        L1(1, 2) = -p2d_2[(n * 2) + 0];

        ls_matrix(Eigen::seqN(0 * 2, 2), Eigen::all) = L0 * P0;
        ls_matrix(Eigen::seqN(1 * 2, 2), Eigen::all) = L1 * P1;

        XYZW = ls_matrix.bdcSvd(Eigen::ComputeFullV).matrixV().col(3); // Previously BDC SVD, Full = Thin

        p3h.col(n) = XYZW / XYZW(3);
    }
}

// OK
static void R_t_from_E(Eigen::Ref<const Eigen::Matrix<float, 3, 3>> const& E, float const* p2d_1, float const* p2d_2, Eigen::Ref<Eigen::Matrix<float, 3, 4>> P, Eigen::Ref<Eigen::Matrix<float, 4, 7>> p3h)
{
    Eigen::Matrix<float, 3, 3> W
    {
        {0.0f, -1.0f, 0.0f},
        {1.0f,  0.0f, 0.0f},
        {0.0f,  0.0f, 1.0f},
    };

    Eigen::BDCSVD<Eigen::Matrix<float, 3, 3>> E_svd = E.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV); // Full = Thin
    Eigen::Matrix<float, 3, 3> U  = E_svd.matrixU();
    Eigen::Matrix<float, 3, 3> Vt = E_svd.matrixV().transpose();

    Eigen::Matrix<float, 3, 3> R1 = U * W             * Vt;
    Eigen::Matrix<float, 3, 3> R2 = U * W.transpose() * Vt;

    if (R1.determinant() < 0) { R1 = -R1; } // Self assign
    if (R2.determinant() < 0) { R2 = -R2; } // Self assign

    Eigen::Matrix<float, 3, 1> pt = U.col(2);
    Eigen::Matrix<float, 3, 1> nt = -pt;

    Eigen::Matrix<float, 3, 4> P0 = Eigen::Matrix<float, 3, 4>::Identity();
    Eigen::Matrix<float, 3, 4> P1;
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

        triangulate(P0, P1, p2d_1, p2d_2, XYZW); // OK

        int64_t count = ((P0 * XYZW).row(2).array() > 0).count() + ((P1 * XYZW).row(2).array() > 0).count();
        if (count < max_count) { continue; }
        max_count = count;

        P = P1;
        p3h = XYZW;
    }
}

// OK
static float R_t_from_TFT(Eigen::Ref<const Eigen::Matrix<float, 27, 1>> const& TFT, float const* p2d_1, float const* p2d_2, float const* p2d_3, Eigen::Ref<Eigen::Matrix<float, 3, 4>> c1, Eigen::Ref<Eigen::Matrix<float, 3, 4>> c2)
{
    Eigen::Matrix<float, 3, 2> e;

    epipoles_from_TFT(TFT, e); // OK

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

    float scale = num / den; // TODO: Mean or Median?

    c2.col(3) = scale * c2.col(3); // Self assign

    return scale;
}

static float compute_scale(float const* p2d, float const* p3d, Eigen::Ref<const Eigen::Matrix<float, 3, 4>> const& c1)
{
    Eigen::Matrix<float, 3, 3> R = c1(Eigen::all, Eigen::seqN(0, 3));
    //Eigen::Matrix<float, 3, 3> Ri = c1(Eigen::all, Eigen::seqN(0, 3)).transpose();
    //Eigen::Matrix<float, 3, 1> ti = -(Ri * c1.col(3));
    Eigen::Matrix<float, 3, 1> P3;
    Eigen::Matrix<float, 3, 1> X3;
    Eigen::Matrix<float, 3, 1> p3;
    Eigen::Matrix<float, 3, 1> p3_X3;
    Eigen::Matrix<float, 3, 1> p3_t3;

    float scales[7];
    float ws;
    int valid = 0;

    float num = 0;
    float den = 0;

    for (int i = 0; i < 7; ++i)
    {
        P3(0) = p3d[(3 * i) + 0];
        P3(1) = p3d[(3 * i) + 1];
        P3(2) = p3d[(3 * i) + 2];

        X3 = R * P3;

        p3(0) = p2d[(2 * i) + 0];
        p3(1) = p2d[(2 * i) + 1];
        p3(2) = 1.0f;

        p3_X3 = p3.cross(X3);
        p3_t3 = p3.cross(c1.col(3));

        num -= p3_X3.dot(p3_t3);
        den += p3_t3.dot(p3_t3);
    }

    return num / den;
}

// OK
/*
static float compute_scale(float const* p2d, float const* p3d, Eigen::Ref<const Eigen::Matrix<float, 3, 4>> const& c1)
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

    std::sort(scales, scales + valid); // TODO: Mean or Median?
    return scales[valid / 2];
}
*/

// OK
static bool trifocal_R_t(float const* p2d_1, float const* p2d_2, float const* p2d_3, float const* sp2d, float const* sp3d, float* tft, float* r1, float* t1, float* r2, float* t2, float* s1, float* s2)
{
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(27, 4 * 7);
    Eigen::Matrix<float, 27, 1> TFT;
    Eigen::Matrix<float, 3, 4> P2;
    Eigen::Matrix<float, 3, 4> P3;

    Eigen::Matrix<float, 2, 7> p2d_1_d;
    Eigen::Matrix<float, 2, 7> p2d_2_d;
    Eigen::Matrix<float, 2, 7> p2d_3_d;

    Eigen::Matrix<float, 2, 7> p2d_1_n;
    Eigen::Matrix<float, 2, 7> p2d_2_n;
    Eigen::Matrix<float, 2, 7> p2d_3_n;

    Eigen::Matrix<float, 3, 3> M1;
    Eigen::Matrix<float, 3, 3> M2;
    Eigen::Matrix<float, 3, 3> M3;

    memcpy(p2d_1_d.data(), p2d_1, 2 * 7 * sizeof(float));
    memcpy(p2d_2_d.data(), p2d_2, 2 * 7 * sizeof(float));
    memcpy(p2d_3_d.data(), p2d_3, 2 * 7 * sizeof(float));

    normalize_points(p2d_1_d, p2d_1_n, M1);
    normalize_points(p2d_2_d, p2d_2_n, M2);
    normalize_points(p2d_3_d, p2d_3_n, M3);

    //build_A(p2d_1, p2d_2, p2d_3, A); // OK
    build_A(p2d_1_n.data(), p2d_2_n.data(), p2d_3_n.data(), A); // OK
    linear_TFT(A, TFT); // OK

    transform_TFT(TFT, M1, M2, M3, TFT, true);

    float local_scale = R_t_from_TFT(TFT, p2d_1, p2d_2, p2d_3, P2, P3); // OK
    float world_scale = (sp2d && sp3d) ? compute_scale(sp2d, sp3d, P2) : 1.0f;

    if (tft) { memcpy(tft, TFT.data(), 27 * sizeof(float)); }

    Eigen::Matrix<float, 3, 3> R2 = P2(Eigen::all, Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 3> R3 = P3(Eigen::all, Eigen::seqN(0, 3));

    Eigen::AngleAxis<float> R01(R2);
    Eigen::AngleAxis<float> R02(R3);

    Eigen::Matrix<float, 3, 1> r01 = R01.angle() * R01.axis();
    Eigen::Matrix<float, 3, 1> r02 = R02.angle() * R02.axis();

    Eigen::Matrix<float, 3, 1> t01 = world_scale * P2.col(3);
    Eigen::Matrix<float, 3, 1> t02 = world_scale * P3.col(3);

    memcpy(r1, r01.data(), 3 * sizeof(float));
    memcpy(r2, r02.data(), 3 * sizeof(float));
    memcpy(t1, t01.data(), 3 * sizeof(float));
    memcpy(t2, t02.data(), 3 * sizeof(float));

    if (s1) { *s1 = world_scale; }
    if (s2) { *s2 = local_scale; }

    return std::isfinite(r1[0] + r1[1] + r1[2] + t1[0] + t1[1] + t1[2] + r2[0] + r2[1] + r2[2] + t2[0] + t2[1] + t2[2]);
}

// OK
bool trifocal_R_t(float const* p2d_1, float const* p2d_2, float const* p2d_3, float const* sp2d, float const* sp3d, float* tft, float* rt1, float* rt2, float* s1, float* s2)
{
    return trifocal_R_t(p2d_1, p2d_2, p2d_3, sp2d, sp3d, tft, rt1 + 0, rt1 + 3, rt2 + 0, rt2 + 3, s1, s2);
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
