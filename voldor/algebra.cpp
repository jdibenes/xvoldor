
#include <complex>

static float refine_root_quadratic(float c, float b, float a, float root, int iterations)
{
    float x = root;

    for (int i = 0; i < iterations; ++i)
    {
    float x2 = x * x;
    
    x -= (a * x2 + b * x + c) / (2 * a * x + b);
    }

    return x;
}

void solve_quadratic(float const* factors, float* real_roots, int refine_iterations)
{
    float const a = factors[2];
    float const b = factors[1];
    float const c = factors[0];

    float d = (b * b) - (4.0f * a * c);
    float s = (d > 0.0f) ? std::sqrt(d) : 0.0f;
    float q = 2.0f * a;

    real_roots[0] = refine_root_quadratic(c, b, a, ((-b + s) / q), refine_iterations);
    real_roots[1] = refine_root_quadratic(c, b, a, ((-b - s) / q), refine_iterations);
}

static float refine_root_quartic(float a0, float a1, float a2, float a3, float a4, float root, int iterations)
{
    float x = root;

    for (int i = 0; i < iterations; ++i)
    {
    float x2 = x * x;
    float x3 = x * x2;
    float x4 = x * x3;

    x -= (a4 * x4 + a3 * x3 + a2 * x2 + a1 * x + a0) / (4 * a4 * x3 + 3 * a3 * x2 + 2 * a2 * x + a1);
    }

    return x;
}

void solve_quartic(float const* factors, float* real_roots, int refine_iterations)
{
    float const a4 = factors[4];
    float const a3 = factors[3];
    float const a2 = factors[2];
    float const a1 = factors[1];
    float const a0 = factors[0];

    float a4_2 = a4 * a4;
    float a3_2 = a3 * a3;
    float a4_3 = a4 * a4_2;
    float a2a4 = a2 * a4;

    float p4 = (8 * a2a4 - 3 * a3_2) / (8 * a4_2);
    float q4 = (a3_2 * a3 - 4 * a2a4 * a3 + 8 * a1 * a4_2) / (8 * a4_3);
    float r4 = (256 * a0 * a4_3 - 3 * (a3_2 * a3_2) - 64 * a1 * a3 * a4_2 + 16 * a2a4 * a3_2) / (256 * (a4_3 * a4));

    float p3 = ((p4 * p4) / 12 + r4) / 3;
    float q3 = (72 * r4 * p4 - 2 * p4 * p4 * p4 - 27 * q4 * q4) / 432;

    float t;

    std::complex<float> w = std::complex<float>(q3 * q3 - p3 * p3 * p3, 0);

    w = std::sqrt(w);

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
    else
    {
        w = std::pow(w, (1.0f / 3.0f));
        t = 4.0f * w.real();
    }

    std::complex<float> sqrt_2m = std::sqrt(std::complex<float>(-2 * p4 / 3 + t, 0));
    
    std::complex<float> complex1 = std::complex<float>(4 * p4 / 3 + t, 0);
    std::complex<float> complex2 = std::complex<float>(2 * q4, 0) / sqrt_2m;

    float B_4A = -a3 / (4 * a4);
    float sqrt_2m_rh = sqrt_2m.real() * 0.5f;

    float sqrt1 = std::sqrt(-(complex1 + complex2)).real() * 0.5f;
    float sqrt2 = std::sqrt(-(complex1 - complex2)).real() * 0.5f;
    
    real_roots[0] = refine_root_quartic(a0, a1, a2, a3, a4, B_4A + sqrt_2m_rh + sqrt1, refine_iterations);
    real_roots[1] = refine_root_quartic(a0, a1, a2, a3, a4, B_4A + sqrt_2m_rh - sqrt1, refine_iterations);
    real_roots[2] = refine_root_quartic(a0, a1, a2, a3, a4, B_4A - sqrt_2m_rh + sqrt2, refine_iterations);
    real_roots[3] = refine_root_quartic(a0, a1, a2, a3, a4, B_4A - sqrt_2m_rh - sqrt2, refine_iterations);
}
