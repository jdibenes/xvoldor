
#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <type_traits>
#include "helpers_traits.h"

using monomial_index_t = int;
using monomial_indices_t = std::vector<monomial_index_t>;

template <typename _scalar, int _n>
class polynomial
{
public:
    using data_t = typename add_vector<_scalar, _n>::type;
    using callback_t = void(_scalar&, monomial_indices_t const&);
    using const_callback_t = void(_scalar const&, monomial_indices_t const&);
    using polynomial_t = polynomial<_scalar, _n>;

private:
    data_t data;
    _scalar zero = _scalar(0);

    template <typename _unpacked, typename _callback>
    void for_each(_unpacked& object, int level, monomial_indices_t& indices, _callback callback)
    {
        if constexpr (std::is_same_v<_unpacked, _scalar>)
        {
            callback(object, indices);
        }
        else
        {
            for (monomial_index_t i = 0; i < object.size(); ++i)
            {
                indices[level] = i;
                for_each(object[i], level + 1, indices, callback);
            }
        }
    }

    template <typename _unpacked, typename _callback>
    void for_each(_unpacked const& object, int level, monomial_indices_t& indices, _callback callback) const
    {
        if constexpr (std::is_same_v<_unpacked, _scalar>)
        {
            callback(object, indices);
        }
        else
        {
            for (monomial_index_t i = 0; i < object.size(); ++i)
            {
                indices[level] = i;
                for_each(object[i], level + 1, indices, callback);
            }
        }
    }

    template <typename _unpacked>
    auto& at(_unpacked& object, int level, monomial_indices_t const& indices)
    {
        if constexpr (std::is_same_v<_unpacked, _scalar>)
        {
            return object;
        }
        else
        {
            monomial_index_t index = indices[level];
            if (index >= object.size()) { object.resize(index + 1); }
            return at(object[index], level + 1, indices);
        }
    }

    template <typename _unpacked>
    auto const& at(_unpacked const& object, int level, monomial_indices_t const& indices) const
    {
        if constexpr (std::is_same_v<_unpacked, _scalar>)
        {
            return object;
        }
        else
        {
            monomial_index_t index = indices[level];
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
        (*this)[monomial_indices_t(_n)] = bias;
    }

    template <typename _callback>
    void for_each(_callback callback)
    {
        monomial_indices_t scratch(_n);
        for_each(data, 0, scratch, callback);
    }

    template <typename _callback>
    void for_each(_callback callback) const
    {
        monomial_indices_t scratch(_n);
        for_each(data, 0, scratch, callback);
    }

    template <typename _other_scalar>
    polynomial<_other_scalar, _n> cast() const
    {
        polynomial<_other_scalar, _n> result;
        auto f = [&](_scalar const& element, monomial_indices_t const& indices) { result[indices] = static_cast<_other_scalar>(element); };
        for_each(f);
        return result;
    }

    auto& operator[](monomial_indices_t const& indices)
    {
        return at(data, 0, indices);
    }

    auto const& operator[](monomial_indices_t const& indices) const
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
        monomial_indices_t indices_c(_n);

        auto f = [&](_scalar const& element_a, monomial_indices_t const& indices_a)
        {
            auto g = [&](_scalar const& element_b, monomial_indices_t const& indices_b)
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

    polynomial_t operator/(_scalar const& other) const
    {
        polynomial_t result = *this;
        return result /= other;
    }

    polynomial_t& operator+=(polynomial_t const& other)
    {
        auto f = [&](_scalar const& element, monomial_indices_t const& indices) { (*this)[indices] += element; };
        other.for_each(f);
        return *this;
    }

    polynomial_t& operator-=(polynomial_t const& other)
    {
        auto f = [&](_scalar const& element, monomial_indices_t const& indices) { (*this)[indices] -= element; };
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
        auto f = [&](_scalar& element, monomial_indices_t const&) { element *= other; };
        for_each(f);
        return *this;
    }

    polynomial_t& operator/=(_scalar const& other)
    {
        auto f = [&](_scalar& element, monomial_indices_t const&) { element /= other; };
        for_each(f);
        return *this;
    }
};

template <int _n>
class grevlex_generator
{
private:
    monomial_indices_t indices;
    monomial_index_t power;
    monomial_index_t sum;
    int index;

    grevlex_generator(monomial_indices_t const& start_indices, int start_index) : indices(_n)
    {
        sum = 0;
        for (int i = 0; i < _n; ++i) { sum += (indices[i] = start_indices[i]); }
        power = sum;
        index = start_index;
    }

public:
    grevlex_generator() : indices(_n), power{ -1 }, sum{ 0 }, index{ -1 }
    {
    }

    grevlex_generator(int start_index) : grevlex_generator(unravel(start_index), start_index)
    {
    }

    grevlex_generator(monomial_indices_t const& start_indices) : grevlex_generator(start_indices, ravel(start_indices))
    {
    }

    monomial_indices_t const& next()
    {
        do
        {
            if (sum > 0)
            {
                int i;
                for (i = _n - 1; (i >= 0) && (indices[i] <= 0); --i)
                {
                    indices[i] = power;
                    sum += power;
                }
                indices[i]--;
                sum--;
            }
            else
            {
                power++;
                indices[0] = power;
                sum = power;
            }
        }
        while (sum != power);
        index++;
        return indices;
    }

    monomial_indices_t const& previous()
    {
        if (power <= 0) { return indices; }
        do
        {
            if (indices[0] < power)
            {
                int i;
                for (i = _n - 1; (i >= 0) && (indices[i] >= power); --i)
                {
                    indices[i] = 0;
                    sum -= power;
                }
                indices[i]++;
                sum++;
            }
            else
            {
                power--;
                indices[0] = 0;
                sum = 0;
            }
        }
        while (sum != power);
        index--;
        return indices;
    }

    monomial_indices_t const& current_indices()
    {
        return indices;
    }

    monomial_index_t current_power()
    {
        return power;
    }

    int current_index()
    {
        return index;
    }

    static monomial_index_t total_degree(monomial_indices_t const& a)
    {
        int sum = 0;
        for (int i = 0; i < _n; ++i) { sum += a[i]; }
        return sum;
    }

    static bool is_equal(monomial_indices_t const& a, monomial_indices_t const& b)
    {
        int i;
        for (i = 0; (i < _n) && (a[i] == b[i]); ++i);
        return i >= _n;
    }

    static bool compare(monomial_indices_t const& a, monomial_indices_t const& b)
    {
        monomial_index_t ap = total_degree(a);
        monomial_index_t bp = total_degree(b);

        if (ap != bp) { return ap < bp; }
        for (int i = 0; i < _n; ++i) { if (a[i] > b[i]) { return true; } }
        return false;
    }

    static monomial_indices_t unravel(int index)
    {
        grevlex_generator<_n> gg;
        for (int i = 0; i < index; ++i) { gg.next(); }
        return gg.next();
    }

    static int ravel(monomial_indices_t const& indices)
    {
        grevlex_generator<_n> gg;
        while (!is_equal(indices, gg.next()));
        return gg.current_index();
    }
};

template <typename _scalar, int _n, typename _index_type, typename _array_type, typename _callback_size, typename _callback_read>
static polynomial<_scalar, _n> vector_to_polynomial_grevlex(_array_type const& a, _callback_size size, _callback_read read)
{
    polynomial<_scalar, _n> p;
    grevlex_generator<_n> gg;
    for (_index_type i = 0; i < size(a); ++i) { p[gg.next()] = read(a, i); }
    return p;
}

template <typename _scalar, int _n, typename A>
static polynomial<_scalar, _n> vector_to_polynomial_grevlex(std::vector<A> const& grevlex_coefficients)
{
    return vector_to_polynomial_grevlex<_scalar, _n, size_t>(grevlex_coefficients, [](std::vector<A> const& a) { return a.size(); }, [](std::vector<A> const& a, size_t i) { return a[i]; });
}

template <typename _scalar, int _n, typename A>
polynomial<_scalar, _n> vector_to_polynomial_grevlex(Eigen::DenseBase<A> const& grevlex_coefficients)
{
    return vector_to_polynomial_grevlex<_scalar, _n, Eigen::Index>(grevlex_coefficients, [](Eigen::DenseBase<A> const& a) { return a.size(); }, [](Eigen::DenseBase<A> const& a, Eigen::Index i) { return a(i); });
}

template <typename _scalar, int _n, int _output_rows, int _output_cols, typename A>
Eigen::Matrix<polynomial<_scalar, _n>, _output_rows, _output_cols> matrix_to_polynomial_grevlex(Eigen::MatrixBase<A> const& M, int output_rows = _output_rows, int output_cols = _output_cols)
{
    Eigen::Matrix<polynomial<_scalar, _n>, _output_rows, _output_cols> E(output_rows, output_cols);

    for (int i = 0; i < output_cols; ++i)
    {
        for (int j = 0; j < output_rows; ++j)
        {
            E(j, i) = vector_to_polynomial_grevlex<_scalar, _n>(M.row((i * output_rows) + j));
        }
    }

    return E;
}






































/*
template <typename _scalar, int _rows, int _cols, int _n>
void polynomial_transfer(Eigen::Ref<Eigen::Matrix<_scalar, _rows, _cols>> dst, polynomial<_scalar, _n> const& src, int row)
{
    auto f = [&](_scalar const& element, monomial_indices_t const& indices) { dst(row, grevlex_generator<_n>::ravel(indices)) = element; };
    src.for_each(f);
}
*/











/*
grevlex example for 3 variables (x, y, z) up to total degree 3

index indices monomial
    0 [0,0,0] 1

    1 [1,0,0] x
    2 [0,1,0] y
    3 [0,0,1] z

    4 [2,0,0] x^2
    5 [1,1,0] x*y
    6 [1,0,1] x*z
    7 [0,2,0] y^2
    8 [0,1,1] y*z
    9 [0,0,2] z^2

   10 [3,0,0] x^3
   11 [2,1,0] x^2*y
   12 [2,0,1] x^2*z
   13 [1,2,0] x*y^2
   14 [1,1,1] x*y*z
   15 [1,0,2] x*z^2
   16 [0,3,0] y^3
   17 [0,2,1] y^2*z
   18 [0,1,2] y*z^2
   19 [0,0,3] z^3
*/
