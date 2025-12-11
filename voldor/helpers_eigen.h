
#pragma once

#include <Eigen/Eigen>

template <typename _scalar, int _rows, int _cols>
Eigen::Matrix<_scalar, _rows, _cols> matrix_from_buffer(_scalar const* data, int rows = _rows, int cols = _cols)
{
    Eigen::Matrix<_scalar, _rows, _cols> M(rows, cols);
    memcpy(M.data(), data, sizeof(_scalar) * rows * cols);
    return M;
}

template <typename _scalar, int _rows, int _cols>
void matrix_to_buffer(Eigen::Matrix<_scalar, _rows, _cols> const& M, _scalar* data)
{
    Eigen::Index rows = M.rows();
    Eigen::Index cols = M.cols();
    memcpy(data, M.data(), sizeof(_scalar) * rows * cols);
}

template <typename _scalar, int _rows_1, int _cols_1, int _rows_2, int _cols_2>
Eigen::Matrix<_scalar, Eigen::Dynamic, Eigen::Dynamic> vector_convolve(Eigen::Matrix<_scalar, _rows_1, _cols_1> const& o1, Eigen::Matrix<_scalar, _rows_2, _cols_2> const& o2)
{
    Eigen::Index sz1 = o1.size();
    Eigen::Index sz2 = o2.size();

    Eigen::Matrix<_scalar, Eigen::Dynamic, Eigen::Dynamic> rv = Eigen::Matrix<_scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(sz1 + sz2 - 1, 1);

    _scalar* pr = rv.data();

    _scalar const* p1 = o1.data();
    _scalar const* p2 = o2.data();

    for (int i1 = 0; i1 < sz1; ++i1)
    {
    for (int i2 = 0; i2 < sz2; ++i2)
    {
    pr[i1 + i2] += p2[i2] * p1[i1];
    }
    }

    return rv;
}

template <typename _scalar, int _rows_1, int _cols_1, int _rows_2, int _cols_2>
Eigen::Matrix<_scalar, Eigen::Dynamic, Eigen::Dynamic> vector_add_padded(Eigen::Matrix<_scalar, _rows_1, _cols_1> const& o1, Eigen::Matrix<_scalar, _rows_2, _cols_2> const& o2, _scalar s1 = 1, _scalar s2 = 1)
{
    Eigen::Index sz1 = o1.size();
    Eigen::Index sz2 = o2.size();

    Eigen::Matrix<_scalar, Eigen::Dynamic, Eigen::Dynamic> rv = Eigen::Matrix<_scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(std::max(sz1, sz2), 1);

    _scalar* pr = rv.data();

    _scalar const* p1 = o1.data();
    _scalar const* p2 = o2.data();

    for (int i1 = 0; i1 < sz1; ++i1) { pr[i1] += s1 * p1[i1]; }
    for (int i2 = 0; i2 < sz2; ++i2) { pr[i2] += s2 * p2[i2]; }

    return rv;
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
