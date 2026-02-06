
#pragma once

#include <Eigen/Eigen>

// OK
template <typename _scalar, int _rows, int _cols>
Eigen::Matrix<_scalar, _rows, _cols> matrix_from_buffer(_scalar const* data, int rows = _rows, int cols = _cols)
{
    Eigen::Matrix<_scalar, _rows, _cols> M(rows, cols);
    memcpy(M.data(), data, sizeof(_scalar) * rows * cols);
    return M;
}

// OK
template <typename _scalar, int _rows, int _cols>
void matrix_to_buffer(Eigen::Matrix<_scalar, _rows, _cols> const& M, _scalar* data)
{
    Eigen::Index rows = M.rows();
    Eigen::Index cols = M.cols();
    memcpy(data, M.data(), sizeof(_scalar) * rows * cols);
}

// OK
template <typename A, typename B>
Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> vector_multiply(Eigen::MatrixBase<A> const& o1, Eigen::MatrixBase<B> const& o2)
{
    Eigen::Index sz1 = o1.size();
    Eigen::Index sz2 = o2.size();

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> rv = Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(sz1 + sz2 - 1, 1);

    for (int i1 = 0; i1 < sz1; ++i1)
    {
    for (int i2 = 0; i2 < sz2; ++i2)
    {
    rv(i1 + i2) += o2(i2) * o1(i1);
    }
    }

    return rv;
}

// OK
template <typename A, typename B>
Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> vector_add(Eigen::MatrixBase<A> const& o1, Eigen::MatrixBase<B> const& o2)
{
    Eigen::Index sz1 = o1.size();
    Eigen::Index sz2 = o2.size();

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> rv = Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(std::max(sz1, sz2), 1);

    for (int i1 = 0; i1 < sz1; ++i1) { rv(i1) += o1(i1); }
    for (int i2 = 0; i2 < sz2; ++i2) { rv(i2) += o2(i2); }

    return rv;
}

// OK
template <typename A, typename B>
Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> vector_subtract(Eigen::MatrixBase<A> const& o1, Eigen::MatrixBase<B> const& o2)
{
    Eigen::Index sz1 = o1.size();
    Eigen::Index sz2 = o2.size();

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> rv = Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(std::max(sz1, sz2), 1);

    for (int i1 = 0; i1 < sz1; ++i1) { rv(i1) += o1(i1); }
    for (int i2 = 0; i2 < sz2; ++i2) { rv(i2) -= o2(i2); }

    return rv;
}
