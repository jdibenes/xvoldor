
#pragma once

#include <Eigen/Eigen>

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

template <typename _scalar, int _rows_1, int _cols_1, int _rows_2, int _cols_2>
Eigen::Matrix<_scalar, _rows_1, _cols_1> cross_matrix(Eigen::Matrix<_scalar, _rows_2, _cols_2> const& v)
{
    Eigen::Matrix<_scalar, _rows_1, _cols_1> M(3, 3);

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

template <typename _scalar, int _rows, int _cols>
Eigen::Matrix<_scalar, _rows, _cols> normalize_essential(Eigen::Matrix<_scalar, _rows, _cols> const& E)
{
    Eigen::JacobiSVD<Eigen::Matrix<_scalar, _rows, _cols>> E_svd = E.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    return E_svd.matrixU() * Eigen::Matrix<_scalar, _rows, _cols>{ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0 } } * E_svd.matrixV().transpose();
}
