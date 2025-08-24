
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h>


template <typename T>
using MatrixA = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign | Eigen::RowMajor>;

template <typename T>
void build_A(T const* points_2D, int count, T* A)
{
    for (int i = 0; i < count; ++i)
    {
        T x1 = points_2D[(i * 6) + 0];
        T y1 = points_2D[(i * 6) + 1];
        T x2 = points_2D[(i * 6) + 2];
        T y2 = points_2D[(i * 6) + 3];
        T x3 = points_2D[(i * 6) + 4];
        T y3 = points_2D[(i * 6) + 5];

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

template <typename T>
void epipoles_from_TFT(T const* TFT, T* e)
{
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd_t1 = Eigen::Map<const Eigen::Matrix<T, 3, 3>>(TFT +  0).jacobiSvd(Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd_t2 = Eigen::Map<const Eigen::Matrix<T, 3, 3>>(TFT +  9).jacobiSvd(Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd_t3 = Eigen::Map<const Eigen::Matrix<T, 3, 3>>(TFT + 18).jacobiSvd(Eigen::ComputeFullV | Eigen::ComputeFullU);

    Eigen::Matrix<T, 3, 3> vx;
    vx << ( svd_t1.matrixV().col(2)),
          ( svd_t2.matrixV().col(2)),
          ( svd_t3.matrixV().col(2));

    Eigen::Matrix<T, 3, 3> ux;
    ux << (-svd_t1.matrixU().col(2)),
          (-svd_t2.matrixU().col(2)),
          (-svd_t3.matrixU().col(2));

    Eigen::Map<Eigen::Matrix<T, 3, 2>> e_m(e);
    e_m << (-ux.jacobiSvd(Eigen::ComputeFullU).matrixU().col(2)),
           (-vx.jacobiSvd(Eigen::ComputeFullU).matrixU().col(2));
}

template <typename T>
void linear_TFT(T const* A, int rows, T* TFT, T threshold = 0)
{
    Eigen::Map<const MatrixA<T>> A_m(A, rows, 27);
    Eigen::Matrix<T, 27, 1> t = A_m.bdcSvd(Eigen::ComputeFullV).matrixV().col(26);

    Eigen::Matrix<T, 3, 2> e;
    epipoles_from_TFT(t.data(), e.data());

    Eigen::Matrix<T, 3, 3> I3 = Eigen::Matrix<T, 3, 3>::Identity();
    Eigen::Matrix<T, 9, 9> I9 = Eigen::Matrix<T, 9, 9>::Identity();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> E(27, 18);
    E << Eigen::kroneckerProduct(I3, Eigen::kroneckerProduct(e.col(1), I3)), Eigen::kroneckerProduct(I9, -e.col(0));

    Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd_E = E.bdcSvd(Eigen::ComputeFullU);
    if (threshold > 0) { svd_E.setThreshold(threshold); }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Up = svd_E.matrixU()(Eigen::all, Eigen::seq(0, svd_E.rank() - 1));

    Eigen::Map<Eigen::Matrix<T, 27, 1>> result(TFT);
    result = Up * ((A_m * Up).bdcSvd(Eigen::ComputeFullV).matrixV()(Eigen::all, Eigen::last));
}

template <typename T>
void cross_matrix(T const* v, T* M)
{
    Eigen::Map<Eigen::Matrix<T, 3, 3>> M_m(M);
    M_m <<     0,   (-v[2]), ( v[1]),
           ( v[2]),     0,   (-v[0]),
           (-v[1]), ( v[0]),      0;
}

template <typename T>
void triangulate(T const* cameras, int count_cameras, T const* points_2D, int count_points, T* points_H3D)
{
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> P_m(     cameras,                 3, 4 * count_cameras);
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> p2D_m( points_2D, 2 * count_cameras, count_points);
    Eigen::Map<      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> h3D_m(points_H3D,                 4, count_points);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ls_matrix(2 * count_cameras, 4);
    Eigen::Matrix<T, 2, 3> L{
        {0, -1, 0.0},
        {1,  0, 0.0},
    };

    for (int n = 0; n < count_points; ++n)
    {
        for (int i = 0; i < count_cameras; ++i)
        {
            L(0, 2) =  p2D_m((i * 2) + 1, n);
            L(1, 2) = -p2D_m((i * 2) + 0, n);
            ls_matrix(Eigen::seq((i * 2) + 0, (i * 2) + 1), Eigen::all) = L * P_m(Eigen::all, Eigen::seq((i * 4) + 0, (i * 4) + 3));
        }
        h3D_m.col(n) = ls_matrix.bdcSvd(Eigen::ComputeFullV).matrixV().col(3);
    }
}

template <typename T>
void
R_t_from_E
(
    T const* E,
    T const* points_2D,
    int count_points,
    T* P//,
    //T* points_3D // TODO
)
{
    Eigen::Matrix<T, 3, 3> W{
        {0, -1, 0},
        {1,  0, 0},
        {0,  0, 1},
    };

    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> E_svd = Eigen::Map<const Eigen::Matrix<T, 3, 3>>(E).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<T, 3, 3> U = E_svd.matrixU();
    Eigen::Matrix<T, 3, 3> Vt = E_svd.matrixV().transpose();

    Eigen::Matrix<T, 3, 3> R1 = U * W * Vt;
    Eigen::Matrix<T, 3, 3> R2 = U * W.transpose() * Vt;
    if (R1.determinant() < 0) { R1 = -R1; }
    if (R2.determinant() < 0) { R2 = -R2; }

    Eigen::Matrix<T, 3, 1> pt = U.col(2);
    Eigen::Matrix<T, 3, 1> nt = -pt;

    Eigen::Matrix<T, 3, 4> P0 = Eigen::Matrix<T, 3, 4>::Identity();
    Eigen::Matrix<T, 3, 4 * 2 * 4> cameras;
    cameras << P0, R1, pt, P0, R1, nt, P0, R2, pt, P0, R2, nt;





    std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> XYZW_1 = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(4, count_points);
    std::shared_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> XYZW_x;
































    int64_t max_count = 0;
    int64_t select = 0;








    for (int i = 0; i < 4; ++i)
    {
        triangulate(cameras.data() + (i * 3 * 4 * 2), 2, points_2D, count_points, XYZW_1.get()->data());




        int64_t count = (XYZW_1.get()->colwise().hnormalized().row(2).array() > 0).count() + ((cameras(Eigen::all, Eigen::seq((i * 8) + 4, (i * 8) + 7)) * *XYZW_1).row(2).array() > 0).count();


        if (count < max_count) { continue; }
        max_count = count;
        select = i;
    }


    Eigen::Map<Eigen::Matrix<T, 3, 4>> P_m(P);
    P_m = cameras(Eigen::all, Eigen::seq((select * 8) + 4, (select * 8) + 7));
}






template <typename T>
void R_t_from_TFT(T const* TFT, T const* points_2D, int count_points, T* Rt01, T* Rt02)
{
    Eigen::Matrix<T, 3, 2> e;
    epipoles_from_TFT(TFT, e.data());

    if (e(2, 0) < 0) { e.col(0) = -e.col(0); }
    if (e(2, 1) < 0) { e.col(1) = -e.col(1); }

    Eigen::Matrix<T, 3, 3> epi21_x;
    Eigen::Matrix<T, 3, 3> epi31_x;
    cross_matrix(e.data() + 0, epi21_x.data());
    cross_matrix(e.data() + 3, epi31_x.data());

    Eigen::Map<const Eigen::Matrix<T, 3, 3>> T1(TFT + 0);
    Eigen::Map<const Eigen::Matrix<T, 3, 3>> T2(TFT + 9);
    Eigen::Map<const Eigen::Matrix<T, 3, 3>> T3(TFT + 18);

    Eigen::Matrix<T, 3, 3> E21;
    Eigen::Matrix<T, 3, 3> E31;

    E21 << (epi21_x * (T1 *             e.col(1))), (epi21_x * (T2 *             e.col(1))), (epi21_x * (T3 *             e.col(1)));
    E31 << (epi31_x * (T1.transpose() * e.col(0))), (epi31_x * (T2.transpose() * e.col(0))), (epi31_x * (T3.transpose() * e.col(0)));
    E31 = -E31;







    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> p2D(points_2D, 6, count_points);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> p21(4, count_points);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> p31(4, count_points);

    p21 << p2D(Eigen::seq(0, 1), Eigen::all), p2D(Eigen::seq(2, 3), Eigen::all);
    p31 << p2D(Eigen::seq(0, 1), Eigen::all), p2D(Eigen::seq(4, 5), Eigen::all);

    R_t_from_E(E21.data(), p21.data(), count_points, Rt01);
    R_t_from_E(E31.data(), p31.data(), count_points, Rt02);







    Eigen::Matrix<T, 3, 4> c0 = Eigen::Matrix<T, 3, 4>::Identity();
    Eigen::Map<Eigen::Matrix<T, 3, 4>> c1(Rt01);
    Eigen::Map<Eigen::Matrix<T, 3, 4>> c2(Rt02);
    Eigen::Matrix<T, 3, 4 * 2> cameras;
    cameras << c0, c1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> p3D(4, count_points);

    triangulate(cameras.data(), 2, p21.data(), count_points, p3D.data());

    Eigen::Matrix<T, 3, Eigen::Dynamic> X(3, count_points);
    Eigen::Matrix<T, 3, Eigen::Dynamic> X3(3, count_points);
    X = p3D.colwise().hnormalized();
    X3 = c2(Eigen::all, Eigen::seq(0, 2)) * X;

    Eigen::Matrix<T, 3, Eigen::Dynamic> p3(3, count_points);
    p3 << p2D(Eigen::seq(4, 5), Eigen::all).colwise().homogeneous();

    Eigen::Matrix<T, 3, Eigen::Dynamic> p3_X3(3, count_points);
    Eigen::Matrix<T, 3, Eigen::Dynamic> p3_t3(3, count_points);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X3dt3(1, count_points);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> p3dt3(1, count_points);
    T numerator = 0;
    T denominator = 0;

    for (int i = 0; i < count_points; ++i)
    {
        p3_X3.col(i) = p3.col(i).cross(X3.col(i));
        p3_t3.col(i) = p3.col(i).cross(c2.col(3));
        X3dt3(0, i) = p3_X3.col(i).dot(p3_t3.col(i));
        p3dt3(0, i) = p3_t3.col(i).dot(p3_t3.col(i));
        numerator -= X3dt3(0, i);
        denominator += p3dt3(0, i);
    }

    c2.col(3) = (numerator / denominator) * c2.col(3);
}








template <typename T>
T compute_scale(T const* points_2D, T const* points_3D, int count, T* RT01, T*)
{
    float* scales = new float[count];

    for (int i = 0; i < count; ++i)
    {
        T X1 = points_3D[(3 * i) + 0];
        T Y1 = points_3D[(3 * i) + 1];
        T Z1 = points_3D[(3 * i) + 2];
        //T x2 = points_2D[(6 * i) + 2];
        //T y2 = points_2D[(6 * i) + 3];
        T x2 = points_2D[(2 * i) + 0];
        T y2 = points_2D[(2 * i) + 1];
        T R_x2 = RT01[0] * x2 + RT01[1] * y2 + RT01[2];
        T R_y2 = RT01[3] * x2 + RT01[4] * y2 + RT01[5];
        T R_w2 = RT01[6] * x2 + RT01[7] * y2 + RT01[8];
        T ntx = -RT01[9];
        T nty = -RT01[10];
        T ntz = -RT01[11];
        T R_ntx = RT01[0] * ntx + RT01[1] * nty + RT01[2] * ntz;
        T R_nty = RT01[3] * ntx + RT01[4] * nty + RT01[5] * ntz;
        T R_ntz = RT01[6] * ntx + RT01[7] * nty + RT01[8] * ntz;
        T tR2_x = R_nty * R_w2 - R_ntz * R_y2;
        T tR2_y = R_ntz * R_x2 - R_ntx * R_w2;
        T tR2_z = R_ntx * R_y2 - R_nty * R_x2;
        T PR2_x = Y1 * R_w2 - Z1 * R_y2;
        T PR2_y = Z1 * R_x2 - X1 * R_w2;
        T PR2_z = X1 * R_y2 - Y1 * R_x2;
        T num2 = PR2_x * PR2_x + PR2_y * PR2_y + PR2_z * PR2_z;
        T den2 = tR2_x * tR2_x + tR2_y * tR2_y + tR2_z * tR2_z; // TODO: if zero??
        T rho = sqrt(num2 / den2);
        scales[i] = rho;
    }

    std::sort(scales, scales + count);
    return scales[count / 2]; // TODO: median?
}






void compute_TFT(float const* points_2D, int count, float const* fx, float const* fy, float const* cx, float const* cy, float* const base_2D, float* const points_3D, float* out_TFT, float* RT01, float* RT02)
{
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> points_nc(6, count);
    float* points_nc_base = points_nc.data();
    float* base_nc = new float[2 * count]; // delete

    for (int i = 0; i < count; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            points_nc_base[(i * 6) + (j * 2) + 0] = (points_2D[(i * 6) + (j * 2) + 0] - cx[j]) / fx[j];
            points_nc_base[(i * 6) + (j * 2) + 1] = (points_2D[(i * 6) + (j * 2) + 1] - cy[j]) / fy[j];
        }
        base_nc[(2 * i) + 0] = (base_2D[(2 * i) + 0] - cx[1]) / fx[1];
        base_nc[(2 * i) + 1] = (base_2D[(2 * i) + 1] - cy[1]) / fy[1];
    }

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign | Eigen::RowMajor> A(4 * count, 27);
    Eigen::Matrix<float, 27, 1> TFT;
    Eigen::Matrix<float, 3, 4> P2;
    Eigen::Matrix<float, 3, 4> P3;

    build_A(points_nc_base, count, A.data());
    linear_TFT(A.data(), 4 * count, TFT.data());
    R_t_from_TFT(TFT.data(), points_nc_base, count, P2.data(), P3.data());
    float scale = compute_scale(base_nc, points_3D, count, P2.data(), P3.data());

    memcpy(out_TFT, TFT.data(), 27 * sizeof(float));
    memcpy(RT01, P2.data(), 3 * 4 * sizeof(float));
    memcpy(RT02, P3.data(), 3 * 4 * sizeof(float));

    RT01[9] *= scale;
    RT01[10] *= scale;
    RT01[11] *= scale;
    RT02[9] *= scale;
    RT02[10] *= scale;
    RT02[11] *= scale;

    delete[]base_nc;
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

