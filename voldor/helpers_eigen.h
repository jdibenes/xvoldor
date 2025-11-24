
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
    int rows = M.rows();
    int cols = M.cols();
    memcpy(data, M.data(), sizeof(_scalar) * rows * cols);
}
