
#include <Eigen/Eigen>
#include <complex>
#include "helpers.h"
#include "helpers_eigen.h"

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

static void solveQuartic(const float* factors, float* realRoots) 
{
    const float& a4 = factors[4];
    const float& a3 = factors[3];
    const float& a2 = factors[2];
    const float& a1 = factors[1];
    const float& a0 = factors[0];

    float a4_2 = a4 * a4;
    float a3_2 = a3 * a3;
    float a4_3 = a4_2 * a4;
    float a2a4 = a2 * a4;

    float p4 = (8 * a2a4 - 3 * a3_2) / (8 * a4_2);
    float q4 = (a3_2 * a3 - 4 * a2a4 * a3 + 8 * a1 * a4_2) / (8 * a4_3);
    float r4 = (256 * a0 * a4_3 - 3 * (a3_2 * a3_2) - 64 * a1 * a3 * a4_2 + 16 * a2a4 * a3_2) / (256 * (a4_3 * a4));

    float p3 = ((p4 * p4) / 12 + r4) / 3; // /=-3
    float q3 = (72 * r4 * p4 - 2 * p4 * p4 * p4 - 27 * q4 * q4) / 432; // /=2

    float t; // *=2

    std::complex<float> w = std::complex<float>(q3 * q3 - p3 * p3 * p3, 0);

    w = std::sqrt(w); //cuCsqrtf(w);
    if (q3 >= 0)
    {
        w.real(-w.real() - q3);
        w.imag(-w.imag());
    }
    else
    {
        w = std::sqrt(w);
        w.real(w.real() - q3);
    }

    if (w.imag() == 0.0f)
    {
        w.real(cbrtf(w.real()));
        t = 2.0f * (w.real() + p3 / w.real());
    }
    else {
        w = std::pow(w, (1.0f / 3.0f));
        t = 4.0f * w.real();
    }

    std::complex<float> sqrt_2m = std::sqrt(std::complex<float>(-2 * p4 / 3 + t, 0));
    float B_4A = -a3 / (4 * a4);
    std::complex<float> complex1 = std::complex<float>(4 * p4 / 3 + t, 0);
    std::complex<float> complex2 = std::complex<float>(2 * q4, 0) / sqrt_2m;

    float sqrt_2m_rh = sqrt_2m.real() * 0.5f;
    float sqrt1 = std::sqrt(-(complex1 + complex2)).real() * 0.5f;
    realRoots[0] = B_4A + sqrt_2m_rh + sqrt1;
    realRoots[1] = B_4A + sqrt_2m_rh - sqrt1;
    float sqrt2 = std::sqrt(-(complex1 - complex2)).real() * 0.5f;
    realRoots[2] = B_4A - sqrt_2m_rh + sqrt2;
    realRoots[3] = B_4A - sqrt_2m_rh - sqrt2;
}


bool solver_gpm_hpc2(float const* pa1, float const* pb1, float const* pc1, float const* pa2, float const* pb2, float const* pc2, float* r01, float* t01)
{
    Eigen::Matrix<float, 3, 1> PA1 = matrix_from_buffer<float, 3, 1>(pa1);
    Eigen::Matrix<float, 3, 1> PB1 = matrix_from_buffer<float, 3, 1>(pb1);
    Eigen::Matrix<float, 3, 1> PC1 = matrix_from_buffer<float, 3, 1>(pc1);
    Eigen::Matrix<float, 3, 1> PA2 = matrix_from_buffer<float, 3, 1>(pa2);
    Eigen::Matrix<float, 3, 1> PB2 = matrix_from_buffer<float, 3, 1>(pb2);
    Eigen::Matrix<float, 3, 1> PC2 = matrix_from_buffer<float, 3, 1>(pc2);

    Eigen::Matrix<float, 3, 1> QB1 = PA1.cross(PB1);
    Eigen::Matrix<float, 3, 1> QC1 = PA1.cross(PC1);

    Eigen::Matrix<float, 3, 1> QB2 = PA2.cross(PB2);
    Eigen::Matrix<float, 3, 1> QC2 = PA2.cross(PC2);

    Eigen::Matrix<float, 10, 1> quadric_1 = solver_gpm_hpc2_build_quadric(QB2, PB1) + solver_gpm_hpc2_build_quadric(PB2, QB1);
    Eigen::Matrix<float, 10, 1> quadric_2 = solver_gpm_hpc2_build_quadric(QC2, PC1) + solver_gpm_hpc2_build_quadric(PC2, QC1);

    Eigen::Matrix<float, 3, 1> DA = PA2 - PA1;

    float sdx = DA(0, 0);
    float sdy = DA(1, 0);
    float sdz = DA(2, 0);

    float udx = std::abs(sdx);
    float udy = std::abs(sdy);
    float udz = std::abs(sdz);

    int index = (udx >= udy) ? ((udx >= udz) ? 0 : 2) : ((udy >= udz) ? 1 : 2);

    Eigen::Matrix<float, 6, 1> conic_1 = solver_gpm_hpc2_build_conic(quadric_1, index, sdx, sdy, sdz);
    Eigen::Matrix<float, 6, 1> conic_2 = solver_gpm_hpc2_build_conic(quadric_1, index, sdx, sdy, sdz);

    Eigen::Matrix<float, 2, 1> ec_1{ -conic_1(4, 0), -conic_1(2, 0) };
    Eigen::Matrix<float, 2, 1> ec_2{  conic_2(4, 0),  conic_2(2, 0) };

    Eigen::Matrix<float, 3, 1> fda_1{ -conic_1(5, 0), -conic_1(3, 0), -conic_1(0, 0) };
    Eigen::Matrix<float, 3, 1> fda_2{ -conic_2(5, 0), -conic_2(3, 0), -conic_2(0, 0) };

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> v2_1 = vector_convolve<float, 2, 1, 3, 1>(ec_2, fda_1);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> v2_2 = vector_convolve<float, 2, 1, 3, 1>(ec_1, fda_2);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> v2 = vector_add_padded(v2_1, v2_2);

    Eigen::Matrix<float, 3, 1> v1 = conic_1(1, 0) * fda_2 - conic_2(1, 0) * fda_1;
    Eigen::Matrix<float, 2, 1> w1 = conic_1(1, 0) *  ec_2 + conic_2(1, 0) *  ec_1;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> lhs = vector_convolve(v1, v1);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> rhs = vector_convolve<float, -1, -1, 2, 1>(v2, w1);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> polynomial = vector_add_padded(lhs, rhs, 1.0f, -1.0f);

    float roots[4];

    solveQuartic(polynomial.data(), roots);

    float kx, ky, kz;
    Eigen::Matrix<float, 3, 3> R;
    Eigen::Matrix<float, 3, 1> t;
    float error = HUGE_VALF;


    for (int r = 0; r < 4; ++r)
    {
        switch (index)
        {
        case 0:  ky = roots[r]; kz = v1.dot(Eigen::Matrix<float, 3, 1>{ 1, ky, ky* ky }) / w1.dot(Eigen::Matrix<float, 2, 1>{ 1, ky* ky }); kx = -((ky * sdy) + (kz * sdz)) / sdx; break;
        case 1:  kx = roots[r]; kz = v1.dot(Eigen::Matrix<float, 3, 1>{ 1, kx, kx* kx }) / w1.dot(Eigen::Matrix<float, 2, 1>{ 1, kx* kx }); ky = -((kx * sdx) + (kz * sdz)) / sdy; break;
        case 2:
        default: kx = roots[r]; ky = v1.dot(Eigen::Matrix<float, 3, 1>{ 1, kx, kx* kx }) / w1.dot(Eigen::Matrix<float, 2, 1>{ 1, kx* kx }); kz = -((kx * sdx) + (ky * sdy)) / sdz; break;
        }

        Eigen::Matrix<float, 3, 3> Ri = Eigen::Matrix<float, 3, 3>
        {
            {1 + kx * kx - ky * ky - kz * kz, 2 * kx * ky - 2 * kz, 2 * kx * kz + 2 * ky},
            {2 * kx * ky + 2 * kz, 1 - kx * kx + ky * ky - kz * kz, 2 * ky * kz - 2 * kx},
            {2 * kx * kz - 2 * ky, 2 * ky * kz + 2 * kx, 1 - kx * kx - ky * ky + kz * kz}
        } / (1 + kx * kx + ky * ky + kz * kz);

        Eigen::Matrix<float, 3, 1> ti = PA2 - Ri * PA1;

        float DB = (PB2 - (Ri * PB1 + ti)).norm();
        float DC = (PC2 - (Ri * PC1 + ti)).norm();

        float e = DB + DC;

        if (e >= error) { continue; }

        R = Ri;
        t = ti;

        error = e;
    }

    matrix_to_buffer(R, r01);
    matrix_to_buffer(t, t01);




    // kx^2
    // ky^2
    // kx*ky
    // ky
    // kx
    // 1
























}


/*
    0 -> 0, 3, 4, 6

    X: 0
    0: 1 0 3
    1: 2 0 4

    X: 3
    X: 4
    2: 5 0 3 4

    X: 6
    3: 7 6
    4: 8 6

    5: 9
    */
    /*
    1 -> 1, 3, 5, 7

    0: 0 1 3
    X: 1
    1: 2 1 5

    X: 3
    2: 4 1 3 5
    X: 5

    3: 6 7
    X: 7
    4: 8 7

    5: 9
    */
    /*
    2 -> 2, 4, 5, 8

    0: 0 2 4
    1: 1 2 5
    X: 2

    2: 3 2 4 5
    X: 4
    X: 5

    3: 6 8
    4: 7 8
    X: 8

    5: 9
    */