
#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <type_traits>
#include "helpers_traits.h"

template <typename _scalar, int _n>
class polynomial
{
public:
    using index_t = int;
    using indices_t = std::vector<index_t>;
    using data_t = typename add_vector<_scalar, _n>::type;
    using callback_t = void(_scalar&, indices_t const&);
    using const_callback_t = void(_scalar const&, indices_t const&);
    using polynomial_t = polynomial<_scalar, _n>;

private:
    data_t data;
    _scalar zero = _scalar(0);

    template <typename _unpacked, typename _callback>
    void for_each(_unpacked& object, int level, indices_t& indices, _callback callback)
    {
        if constexpr (std::is_arithmetic_v<_unpacked>)
        {
            callback(object, indices);
        }
        else
        {
            for (index_t i = 0; i < object.size(); ++i)
            {
                indices[level] = i;
                for_each(object[i], level + 1, indices, callback);
            }
        }
    }

    template <typename _unpacked, typename _callback>
    void for_each(_unpacked const& object, int level, indices_t& indices, _callback callback) const
    {
        if constexpr (std::is_arithmetic_v<_unpacked>)
        {
            callback(object, indices);
        }
        else
        {
            for (index_t i = 0; i < object.size(); ++i)
            {
                indices[level] = i;
                for_each(object[i], level + 1, indices, callback);
            }
        }
    }

    template <typename _unpacked>
    auto& at(_unpacked& object, int level, indices_t const& indices)
    {
        if constexpr (std::is_arithmetic_v<_unpacked>)
        {
            return object;
        }
        else
        {
            index_t index = indices[level];
            if (index >= object.size()) { object.resize(index + 1); }
            return at(object[index], level + 1, indices);
        }
    }

    template <typename _unpacked>
    auto const& at(_unpacked const& object, int level, indices_t const& indices) const
    {
        if constexpr (std::is_arithmetic_v<_unpacked>)
        {
            return object;
        }
        else
        {
            index_t index = indices[level];
            if (index >= object.size()) { return zero; }
            return at(object[index], level + 1, indices);
        }
    }

public:
    polynomial()
    {
    }

    polynomial(_scalar const& bias)
    {
        (*this)[indices_t(_n)] = bias;
    }

    template <typename _callback>
    void for_each(_callback callback)
    {
        indices_t scratch(_n);
        for_each(data, 0, scratch, callback);
    }

    template <typename _callback>
    void for_each(_callback callback) const
    {
        indices_t scratch(_n);
        for_each(data, 0, scratch, callback);
    }

    template <typename _other_scalar>
    polynomial<_other_scalar, _n> cast() const
    {
        polynomial<_other_scalar, _n> result;
        auto f = [&](_scalar const& element, indices_t const& indices) { result[indices] = static_cast<_other_scalar>(element); };
        for_each(f);
        return result;
    }

    auto& operator[](indices_t const& indices)
    {
        return at(data, 0, indices);
    }

    auto const& operator[](indices_t const& indices) const
    {
        return at(data, 0, indices);
    }

    polynomial_t operator+() const
    {
        return *this;
    }

    polynomial_t operator+(polynomial_t const& other) const
    {
        polynomial_t result = *this;
        return result += other;
    }

    polynomial_t operator-() const
    {
        polynomial_t result;
        return result -= *this;
    }

    polynomial_t operator-(polynomial_t const& other) const
    {
        polynomial_t result = *this;
        return result -= other;
    }

    polynomial_t operator*(polynomial_t const& other) const
    {
        polynomial_t result;
        indices_t indices_c(_n);

        auto f = [&](_scalar const& element_a, indices_t const& indices_a)
        {
        auto g = [&](_scalar const& element_b, indices_t const& indices_b)
        {
        for (int i = 0; i < _n; ++i) { indices_c[i] = indices_a[i] + indices_b[i]; }
        result[indices_c] += element_a * element_b;
        };
        other.for_each(g);
        };
        for_each(f);

        return result;
    }

    polynomial_t operator*(_scalar const& other) const
    {
        polynomial_t result = *this;
        return result *= other;
    }

    friend polynomial<_scalar, _n> operator*(_scalar const& other, polynomial<_scalar, _n> const& x)
    {
        return x * other;
    }

    polynomial_t& operator+=(polynomial_t const& other)
    {
        auto f = [&](_scalar const& element, indices_t const& indices) { (*this)[indices] += element; };
        other.for_each(f);
        return *this;
    }

    polynomial_t& operator-=(polynomial_t const& other)
    {
        auto f = [&](_scalar const& element, indices_t const& indices) { (*this)[indices] -= element; };
        other.for_each(f);
        return *this;
    }

    polynomial_t& operator*=(polynomial_t const& other)
    {
        *this = *this * other; // true *= has aliasing issues
        return *this;
    }

    polynomial_t& operator*=(_scalar const& other)
    {
        auto f = [&](_scalar& element, indices_t const&) { element *= other; };
        for_each(f);
        return *this;
    }
};

template <typename _scalar, int _n, int _output_rows, int _output_cols, typename A>
Eigen::Matrix<polynomial<_scalar, _n>, _output_rows, _output_cols> matrix_to_linear_polynomial_matrix(Eigen::MatrixBase<A> const& M, int output_rows = _output_rows, int output_cols = _output_cols)
{
    Eigen::Matrix<polynomial<_scalar, _n>, _output_rows, _output_cols> E(output_rows, output_cols);
    typename polynomial<_scalar, _n>::indices_t indices(_n);

    for (int i = 0; i < output_cols; ++i)
    {
    for (int j = 0; j < output_rows; ++j)
    {
    for (int p = 0; p < (_n + 1); ++p)
    {
    for (int q = 0; q < _n; ++q) { indices[q] = p == (q + 1); }
    E(j, i)[indices] = M((i * output_rows) + j, p);
    }
    }
    }

    return E;
}
