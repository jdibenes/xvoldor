
#pragma once

#include <type_traits>
#include <Eigen/Eigen>

// OK
template <typename _scalar_out, int _rows, int _cols, typename _scalar_in>
Eigen::Matrix<_scalar_out, _rows, _cols> matrix_from_buffer(_scalar_in const* data, int rows = _rows, int cols = _cols)
{
    Eigen::Map<Eigen::Matrix<_scalar_in, _rows, _cols> const> S(data, rows, cols);
    if constexpr (std::is_same_v<_scalar_out, _scalar_in>) { return S; } else { return S.cast<_scalar_out>(); }
}

// OK
template <typename _scalar_in, int _rows, int _cols, typename _scalar_out>
void matrix_to_buffer(Eigen::Matrix<_scalar_in, _rows, _cols> const& M, _scalar_out* data)
{
    Eigen::Map<Eigen::Matrix<_scalar_out, _rows, _cols>> D(data, M.rows(), M.cols());
    if constexpr (std::is_same_v<_scalar_out, _scalar_in>) { D = M; } else { D = M.cast<_scalar_out>(); }
}

// OK
template <typename A, typename B>
Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> vector_multiply(Eigen::MatrixBase<A> const& o1, Eigen::MatrixBase<B> const& o2)
{
    Eigen::Index sz1 = o1.size();
    Eigen::Index sz2 = o2.size();

    Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic> rv = Eigen::Matrix<typename A::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(sz1 + sz2 - 1, 1);

    for (int i1 = 0; i1 < sz1; ++i1) { for (int i2 = 0; i2 < sz2; ++i2) { rv(i1 + i2) += o2(i2) * o1(i1); } }

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
