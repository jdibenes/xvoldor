
//
// TODO: restore NDEBUG 
//


#define EIGEN_NO_AUTOMATIC_RESIZING

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h>

//-----------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------

// OK
void build_A(float const* points_2D, int count, Eigen::Ref<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> result)
{
    float* A = result.data();

    for (int i = 0; i < count; ++i)
    {
        float x1 = points_2D[(i * 6) + 0];
        float y1 = points_2D[(i * 6) + 1];
        float x2 = points_2D[(i * 6) + 2];
        float y2 = points_2D[(i * 6) + 3];
        float x3 = points_2D[(i * 6) + 4];
        float y3 = points_2D[(i * 6) + 5];

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
    M <<     0,   (-v[2]), ( v[1]),
         ( v[2]),     0,   (-v[0]),
         (-v[1]), ( v[0]),     0;
}

// OK
void triangulate(Eigen::Ref<const Eigen::Matrix<float, 3, 4 * 2>> const& cameras, Eigen::Ref<const Eigen::Matrix<float, 4, 7>> const& points_2D, Eigen::Ref<Eigen::Matrix<float, 4, 7>> points_H3D)
{
    Eigen::Matrix<float, 2 * 2, 4> ls_matrix;
    Eigen::Matrix<float, 2, 3> L
    {
        {0.0f, -1.0f, 0.0f},
        {1.0f,  0.0f, 0.0f},
    };

    for (int n = 0; n < 7; ++n)
    {
        for (int i = 0; i < 2; ++i)
        {
            L(0, 2) =  points_2D((i * 2) + 1, n);
            L(1, 2) = -points_2D((i * 2) + 0, n);

            ls_matrix(Eigen::seqN((i * 2) + 0, 2), Eigen::all) = L * cameras(Eigen::all, Eigen::seqN((i * 4) + 0, 4));
        }
        points_H3D.col(n) = ls_matrix.bdcSvd(Eigen::ComputeFullV).matrixV().col(3);
    }
}

// OK
void R_t_from_E(Eigen::Ref<const Eigen::Matrix<float, 3, 3>> const& E, Eigen::Ref<const Eigen::Matrix<float, 4, 7>> const& points_2D, Eigen::Ref<Eigen::Matrix<float, 3, 4>> P, Eigen::Ref<Eigen::Matrix<float, 4, 7>> points_H3D)
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

        triangulate(PX, points_2D, XYZW); // OK

        int64_t count = (XYZW.colwise().hnormalized().row(2).array() > 0).count() + ((P1 * XYZW).row(2).array() > 0).count();
        if (count < max_count) { continue; }
        max_count = count;

        P = P1;
        points_H3D = XYZW;
    }
}

// OK
float R_t_from_TFT(Eigen::Ref<const Eigen::Matrix<float, 27, 1>> const& TFT, Eigen::Ref<const Eigen::Matrix<float, 6, 7>> const& points_2D, Eigen::Ref<Eigen::Matrix<float, 3, 4>> c1, Eigen::Ref<Eigen::Matrix<float, 3, 4>> c2)
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

    Eigen::Matrix<float, 4, 7> p21;
    Eigen::Matrix<float, 4, 7> p31;

    p21 << points_2D(Eigen::seqN(0, 2), Eigen::all), points_2D(Eigen::seqN(2, 2), Eigen::all);
    p31 << points_2D(Eigen::seqN(0, 2), Eigen::all), points_2D(Eigen::seqN(4, 2), Eigen::all);

    Eigen::Matrix<float, 4, 7> p3DH(4, 7);

    R_t_from_E(E31, p31, c2, p3DH); // OK
    R_t_from_E(E21, p21, c1, p3DH); // OK

    Eigen::Matrix<float, 3, 7> X3;
    Eigen::Matrix<float, 3, 7> p3;

    X3 = c2(Eigen::all, Eigen::seqN(0, 3)) * p3DH.colwise().hnormalized();
    p3 = points_2D(Eigen::seqN(4, 2), Eigen::all).colwise().homogeneous();

    Eigen::Matrix<float, 3, 7> p3_X3;
    Eigen::Matrix<float, 3, 7> p3_t3;
    Eigen::Matrix<float, 1, 7> X3dt3;
    Eigen::Matrix<float, 1, 7> p3dt3;

    float num = 0;
    float den = 0;

    for (int i = 0; i < 7; ++i)
    {
        p3_X3.col(i) = p3.col(i).cross(X3.col(i));
        p3_t3.col(i) = p3.col(i).cross(c2.col(3));
        X3dt3(0, i) = p3_X3.col(i).dot(p3_t3.col(i));
        p3dt3(0, i) = p3_t3.col(i).dot(p3_t3.col(i));
        num -= X3dt3(0, i);
        den += p3dt3(0, i);
    }

    float scale = num / den;

    c2.col(3) = scale * c2.col(3);

    return scale;
}







// OK
float compute_scale(float const* points_2D, float const* points_3D, int count, float* RT01)
{
    std::unique_ptr<float[]> scales = std::make_unique<float[]>(count);
    int valid = 0;

    for (int i = 0; i < count; ++i)
    {
        float X1 = points_3D[(3 * i) + 0];
        float Y1 = points_3D[(3 * i) + 1];
        float Z1 = points_3D[(3 * i) + 2];
        float x2 = points_2D[(2 * i) + 0];
        float y2 = points_2D[(2 * i) + 1];
        float R_x2 = RT01[0] * x2 + RT01[1] * y2 + RT01[2];
        float R_y2 = RT01[3] * x2 + RT01[4] * y2 + RT01[5];
        float R_w2 = RT01[6] * x2 + RT01[7] * y2 + RT01[8];
        float ntx = -RT01[9];
        float nty = -RT01[10];
        float ntz = -RT01[11];
        float R_ntx = RT01[0] * ntx + RT01[1] * nty + RT01[2] * ntz;
        float R_nty = RT01[3] * ntx + RT01[4] * nty + RT01[5] * ntz;
        float R_ntz = RT01[6] * ntx + RT01[7] * nty + RT01[8] * ntz;
        float tR2_x = R_nty * R_w2 - R_ntz * R_y2;
        float tR2_y = R_ntz * R_x2 - R_ntx * R_w2;
        float tR2_z = R_ntx * R_y2 - R_nty * R_x2;
        float PR2_x = Y1 * R_w2 - Z1 * R_y2;
        float PR2_y = Z1 * R_x2 - X1 * R_w2;
        float PR2_z = X1 * R_y2 - Y1 * R_x2;
        float num2 = PR2_x * PR2_x + PR2_y * PR2_y + PR2_z * PR2_z;
        float den2 = tR2_x * tR2_x + tR2_y * tR2_y + tR2_z * tR2_z;
        float rho = std::sqrt(num2 / den2);

        if (!std::isfinite(rho)) { continue; }

        scales[valid++] = rho;
    }

    if (valid <= 0) { return 0; }

    // TODO: mean or median or ?
    std::sort(scales.get(), scales.get() + valid);
    return scales[valid / 2];
}












void
trifocal_R_t
(
    float const* points_2D,
    int count,
    float const* fx,
    float const* fy,
    float const* cx,
    float const* cy,
    float* const map_2D,
    float* const map_3D,
    float* out_TFT,
    float* out_r01,
    float* out_t01,
    float* out_r02,
    float* out_t02
)
{
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> points_nc(6, count);
    float* points_nc_base = points_nc.data();
    float* map_nc = new float[2 * count]; // delete

    for (int i = 0; i < count; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            points_nc_base[(i * 6) + (j * 2) + 0] = (points_2D[(i * 6) + (j * 2) + 0] - cx[j]) / fx[j];
            points_nc_base[(i * 6) + (j * 2) + 1] = (points_2D[(i * 6) + (j * 2) + 1] - cy[j]) / fy[j];
        }
        map_nc[(2 * i) + 0] = (map_2D[(2 * i) + 0] - cx[1]) / fx[1];
        map_nc[(2 * i) + 1] = (map_2D[(2 * i) + 1] - cy[1]) / fy[1];
    }






    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(27, 4 * count);
    Eigen::Matrix<float, 27, 1> TFT;
    Eigen::Matrix<float, 3, 4> P2;
    Eigen::Matrix<float, 3, 4> P3;

    build_A(points_nc_base, count, A); // OK
    linear_TFT(A, TFT); // OK
    float local_scale = R_t_from_TFT(TFT, points_nc, P2, P3); // OK
    float scale = compute_scale(map_nc, map_3D, count, P2.data());






    //float scale = 1.0;

    cv::Mat R01(3, 3, CV_32FC1, P2.data()); // needs transpose
    cv::Mat R02(3, 3, CV_32FC1, P3.data()); // needs transpose
    cv::Mat r01(3, 1, CV_32FC1, out_r01);
    cv::Mat r02(3, 1, CV_32FC1, out_r02);

    cv::Rodrigues(R01.t(), r01);
    cv::Rodrigues(R02.t(), r02);

    cv::Mat ut01(3, 1, CV_32FC1, P2.data() + 9);
    cv::Mat ut02(3, 1, CV_32FC1, P3.data() + 9);
    cv::Mat st01(3, 1, CV_32FC1, out_t01);
    cv::Mat st02(3, 1, CV_32FC1, out_t02);

    st01 = scale * ut01;
    st02 = scale * ut02;

    memcpy(out_TFT, TFT.data(), 27 * sizeof(float));
    





    

    delete[] map_nc;
}

void print_TFT(float const* TFT)
{
    std::cout << "[ ";
    for (int i = 0; i < 27; ++i)
    {
        std::cout << TFT[i];
        if (i < 26) { std::cout << ","; }
    }
    std::cout << "]" << std::endl;
}


/*
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> E = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Zero(27, 18);

    float* pE = E.data();

    float e_01 =  e(0, 1);
    float e_11 =  e(1, 1);
    float e_21 =  e(2, 1);
    float e_00 = -e(0, 0);
    float e_10 = -e(1, 0);
    float e_20 = -e(2, 0);

    pE[  0 +   0 +  0 + 0] = e_01;
    pE[  0 +   0 + 27 + 1] = e_01;
    pE[  0 +   0 + 54 + 2] = e_01;
    pE[  0 +   3 +  0 + 0] = e_11;
    pE[  0 +   3 + 27 + 1] = e_11;
    pE[  0 +   3 + 54 + 2] = e_11;
    pE[  0 +   6 +  0 + 0] = e_21;
    pE[  0 +   6 + 27 + 1] = e_21;
    pE[  0 +   6 + 54 + 2] = e_21;
    pE[ 81 +   9 +  0 + 0] = e_01;
    pE[ 81 +   9 + 27 + 1] = e_01;
    pE[ 81 +   9 + 54 + 2] = e_01;
    pE[ 81 +  12 +  0 + 0] = e_11;
    pE[ 81 +  12 + 27 + 1] = e_11;
    pE[ 81 +  12 + 54 + 2] = e_11;
    pE[ 81 +  15 +  0 + 0] = e_21;
    pE[ 81 +  15 + 27 + 1] = e_21;
    pE[ 81 +  15 + 54 + 2] = e_21;
    pE[162 +  18 +  0 + 0] = e_01;
    pE[162 +  18 + 27 + 1] = e_01;
    pE[162 +  18 + 54 + 2] = e_01;
    pE[162 +  21 +  0 + 0] = e_11;
    pE[162 +  21 + 27 + 1] = e_11;
    pE[162 +  21 + 54 + 2] = e_11;
    pE[162 +  24 +  0 + 0] = e_21;
    pE[162 +  24 + 27 + 1] = e_21;
    pE[162 +  24 + 54 + 2] = e_21;
    pE[243 +   0 +  0 + 0] = e_00;
    pE[243 +   0 +  0 + 1] = e_10;
    pE[243 +   0 +  0 + 2] = e_20;
    pE[243 +  27 +  3 + 0] = e_00;
    pE[243 +  27 +  3 + 1] = e_10;
    pE[243 +  27 +  3 + 2] = e_20;
    pE[243 +  54 +  6 + 0] = e_00;
    pE[243 +  54 +  6 + 1] = e_10;
    pE[243 +  54 +  6 + 2] = e_20;
    pE[243 +  81 +  9 + 0] = e_00;
    pE[243 +  81 +  9 + 1] = e_10;
    pE[243 +  81 +  9 + 2] = e_20;
    pE[243 + 108 + 12 + 0] = e_00;
    pE[243 + 108 + 12 + 1] = e_10;
    pE[243 + 108 + 12 + 2] = e_20;
    pE[243 + 135 + 15 + 0] = e_00;
    pE[243 + 135 + 15 + 1] = e_10;
    pE[243 + 135 + 15 + 2] = e_20;
    pE[243 + 162 + 18 + 0] = e_00;
    pE[243 + 162 + 18 + 1] = e_10;
    pE[243 + 162 + 18 + 2] = e_20;
    pE[243 + 189 + 21 + 0] = e_00;
    pE[243 + 189 + 21 + 1] = e_10;
    pE[243 + 189 + 21 + 2] = e_20;
    pE[243 + 216 + 24 + 0] = e_00;
    pE[243 + 216 + 24 + 1] = e_10;
    pE[243 + 216 + 24 + 2] = e_20;
    */