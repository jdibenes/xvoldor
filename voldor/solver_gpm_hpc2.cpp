
#include <Eigen/Eigen>
#include "algebra.h"
#include "helpers.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"

static Eigen::Matrix<float, 10, 1> solver_gpm_hpc2_build_quadric(Eigen::Matrix<float, 3, 1> const& q, Eigen::Matrix<float, 3, 1> const& p)
{
    float qx = q(0, 0);
    float qy = q(1, 0);
    float qz = q(2, 0);

    float px = p(0, 0);
    float py = p(1, 0);
    float pz = p(2, 0);

    Eigen::Matrix<float, 10, 1> quadric;

    quadric(0, 0) = + (qx * px) - (qy * py) - (qz * pz); // kx^2
    quadric(1, 0) = - (qx * px) + (qy * py) - (qz * pz); // ky^2
    quadric(2, 0) = - (qx * px) - (qy * py) + (qz * pz); // kz^2
    quadric(3, 0) = 2.0f * ((qy * px) + (qx * py));      // kx*ky
    quadric(4, 0) = 2.0f * ((qz * px) + (qx * pz));      // kx*kz
    quadric(5, 0) = 2.0f * ((qz * py) + (qy * pz));      // ky*kz
    quadric(6, 0) = 2.0f * ((qz * py) - (qy * pz));      // kx
    quadric(7, 0) = 2.0f * ((qx * pz) - (qz * px));      // ky
    quadric(8, 0) = 2.0f * ((qy * px) - (qx * py));      // kz
    quadric(9, 0) = + (qx * px) + (qy * py) + (qz * pz); // 1

    return quadric;
}

static Eigen::Matrix<float, 6, 1> solver_gpm_hpc2_build_conic(Eigen::Matrix<float, 10, 1> const& quadric, int index, float sdx, float sdy, float sdz)
{
    Eigen::Matrix<float, 6, 1> conic;

    int i0, i1, i2, i3, i4, i5;
    int x0, x1, x2, x3;

    float a, b;

    switch (index)
    {
    case 0:  i0 = 1; i1 = 2; i2 = 5; i3 = 7; i4 = 8; i5 = 9; x0 = 0; x1 = 3; x2 = 4; x3 = 6; a = sdy / sdx; b = sdz / sdx; break;
    case 1:  i0 = 0; i1 = 2; i2 = 4; i3 = 6; i4 = 8; i5 = 9; x0 = 1; x1 = 3; x2 = 5; x3 = 7; a = sdx / sdy; b = sdz / sdy; break;
    case 2:
    default: i0 = 0; i1 = 1; i2 = 3; i3 = 6; i4 = 7; i5 = 9; x0 = 2; x1 = 4; x2 = 5; x3 = 8; a = sdx / sdz; b = sdy / sdz; break;
    }

    conic(0, 0) = quadric(i0, 0) +        quadric(x0, 0) * (a * a) - quadric(x1, 0) * a; // ky^2;
    conic(1, 0) = quadric(i1, 0) +        quadric(x0, 0) * (b * b) - quadric(x2, 0) * b; // kz^2
    conic(2, 0) = quadric(i2, 0) + 2.0f * quadric(x0, 0) * (a * b) - quadric(x1, 0) * b 
                                                                   - quadric(x2, 0) * a; // ky*kz
    conic(3, 0) = quadric(i3, 0)                                   - quadric(x3, 0) * a; // ky
    conic(4, 0) = quadric(i4, 0)                                   - quadric(x3, 0) * b; // kz
    conic(5, 0) = quadric(i5, 0);                                                        // 1

    return conic;
}

bool solver_gpm_hpc2(float const* pa1, float const* pb1, float const* pc1, float const* pa2, float const* pb2, float const* pc2, float* r01, float* t01, int refine_iterations)
{
    Eigen::Matrix<float, 3, 1> PA1 = matrix_from_buffer<float, 3, 1>(pa1);
    Eigen::Matrix<float, 3, 1> PB1 = matrix_from_buffer<float, 3, 1>(pb1);
    Eigen::Matrix<float, 3, 1> PC1 = matrix_from_buffer<float, 3, 1>(pc1);
    Eigen::Matrix<float, 3, 1> PA2 = matrix_from_buffer<float, 3, 1>(pa2);
    Eigen::Matrix<float, 3, 1> PB2 = matrix_from_buffer<float, 3, 1>(pb2);
    Eigen::Matrix<float, 3, 1> PC2 = matrix_from_buffer<float, 3, 1>(pc2);

    Eigen::Matrix<float, 3, 1> QB1 = PA1.cross(PB1);
    Eigen::Matrix<float, 3, 1> QC1 = PA1.cross(PC1);

    Eigen::Matrix<float, 3, 1> QB2 = PB2.cross(PA2);
    Eigen::Matrix<float, 3, 1> QC2 = PC2.cross(PA2);

    Eigen::Matrix<float, 10, 1> quadric_1 = solver_gpm_hpc2_build_quadric(QB2, PB1) - solver_gpm_hpc2_build_quadric(PB2, QB1);
    Eigen::Matrix<float, 10, 1> quadric_2 = solver_gpm_hpc2_build_quadric(QC2, PC1) - solver_gpm_hpc2_build_quadric(PC2, QC1);

    quadric_1.normalize();
    quadric_2.normalize();

    Eigen::Matrix<float, 3, 1> DA = PA2 - PA1;

    DA.normalize();

    float sdx = DA(0, 0);
    float sdy = DA(1, 0);
    float sdz = DA(2, 0);

    float udx = std::abs(sdx);
    float udy = std::abs(sdy);
    float udz = std::abs(sdz);

    int index = (udx >= udy) ? ((udx >= udz) ? 0 : 2) : ((udy >= udz) ? 1 : 2);

    Eigen::Matrix<float, 6, 1> conic_1 = solver_gpm_hpc2_build_conic(quadric_1, index, sdx, sdy, sdz);
    Eigen::Matrix<float, 6, 1> conic_2 = solver_gpm_hpc2_build_conic(quadric_2, index, sdx, sdy, sdz);

    conic_1.normalize();
    conic_2.normalize();

    Eigen::Matrix<float, 3, 1> feb_1{ -conic_1(5, 0), -conic_1(4, 0), -conic_1(1, 0) };
    Eigen::Matrix<float, 3, 1> feb_2{ -conic_2(5, 0), -conic_2(4, 0), -conic_2(1, 0) };

    Eigen::Matrix<float, 2, 1> dc_1{ -conic_1(3, 0), -conic_1(2, 0) };
    Eigen::Matrix<float, 2, 1> dc_2{  conic_2(3, 0),  conic_2(2, 0) };

    float a_1 = conic_1(0, 0);
    float a_2 = conic_2(0, 0);
 
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> v2_1 = vector_convolve(dc_2, feb_1);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> v2_2 = vector_convolve(dc_1, feb_2);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> v2 = vector_add_padded(v2_1, v2_2);

    Eigen::Matrix<float, 3, 1> v1 = a_1 * feb_2 - a_2 * feb_1;
    Eigen::Matrix<float, 2, 1> w1 = a_1 *  dc_2 + a_2 *  dc_1;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> lhs = vector_convolve(v1, v1);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> rhs = vector_convolve(v2, w1);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> polynomial = vector_add_padded(lhs, rhs, 1.0f, -1.0f);

    polynomial.normalize();

    float roots[4];

    solve_quartic(polynomial.data(), roots, refine_iterations);

    float kx;
    float ky;
    float kz;

    Eigen::Matrix<float, 3, 3> R;
    Eigen::Matrix<float, 3, 1> t;

    float error = HUGE_VALF;
    bool  set   = false;

    for (int r = 0; r < 4; ++r)
    {
    switch (index)
    {
    case 0:  kz = roots[r]; ky = v1.dot(Eigen::Matrix<float, 3, 1>{ 1, kz, kz * kz }) / w1.dot(Eigen::Matrix<float, 2, 1>{ 1, kz }); kx = -((ky * sdy) + (kz * sdz)) / sdx; break; // x eliminated, solving for (y,y^2), z is hidden
    case 1:  kz = roots[r]; kx = v1.dot(Eigen::Matrix<float, 3, 1>{ 1, kz, kz * kz }) / w1.dot(Eigen::Matrix<float, 2, 1>{ 1, kz }); ky = -((kx * sdx) + (kz * sdz)) / sdy; break; // y eliminated, solving for (x,x^2), z is hidden
    case 2:
    default: ky = roots[r]; kx = v1.dot(Eigen::Matrix<float, 3, 1>{ 1, ky, ky * ky }) / w1.dot(Eigen::Matrix<float, 2, 1>{ 1, ky }); kz = -((kx * sdx) + (ky * sdy)) / sdz; break; // z eliminated, solving for (x,x^2), y is hidden
    }

    Eigen::Matrix<float, 3, 3> Ri = matrix_R_cayley<float, 3, 3>(kx, ky, kz);
    Eigen::Matrix<float, 3, 1> ti = PA2 - Ri * PA1;

    float DB = (PB2 - (Ri * PB1 + ti)).norm();
    float DC = (PC2 - (Ri * PC1 + ti)).norm();

    float e = DB + DC;

    if (!std::isfinite(e) || (e >= error)) { continue; }

    R = Ri;
    t = ti;

    error = e;
    set   = true;
    }

    if (!set) { return false; }

    Eigen::AngleAxis<float> aa(R);

    Eigen::Matrix<float, 3, 1> r = aa.axis() * aa.angle();

    matrix_to_buffer(r, r01);
    matrix_to_buffer(t, t01);

    float r_sum = r01[0] + r01[1] + r01[2];
    float t_sum = t01[0] + t01[1] + t01[2];

    float x_sum = r_sum + t_sum;

    return std::isfinite(x_sum);
}
