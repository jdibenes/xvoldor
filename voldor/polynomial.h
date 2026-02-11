
#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <type_traits>

using monomial_index_t = int;
using monomial_indices_t = std::vector<monomial_index_t>;
using monomial_indices_layered_t = std::vector<monomial_indices_t>;

template <typename _scalar, int _n>
class polynomial
{
private:
    template <typename T>
    struct remove_polynomial
    {
        typedef T type;
        enum { level = 0 };
        enum { count = 0 };        
    };

    template <typename S, int V>
    struct remove_polynomial<polynomial<S, V>>
    {
        typedef typename remove_polynomial<S>::type type;
        enum { level = 1 + remove_polynomial<S>::level };
        enum { count = V + remove_polynomial<S>::count };        
    };

    template <typename T, int n>
    struct add_vector 
    {
        typedef typename add_vector<std::vector<T>, n - 1>::type type;
    };

    template <typename T>
    struct add_vector<T, 0>
    {
        typedef T type;
    };

public:
    using scalar_t = _scalar;
    enum { variables_n = _n };

    using data_t = typename add_vector<_scalar, _n>::type;
    using polynomial_t = polynomial<_scalar, _n>;
    using arithmetic_t = typename remove_polynomial<polynomial_t>::type;
    enum { layers_n = remove_polynomial<polynomial_t>::level };
    enum { variables_arithmetic_n = remove_polynomial<polynomial_t>::count };
    
    using callback_t = void(_scalar&, monomial_indices_t const&);
    using const_callback_t = void(_scalar const&, monomial_indices_t const&);
    using callback_arithmetic_t = void(arithmetic_t&, monomial_indices_t const&);
    using const_callback_arithmetic_t = void(arithmetic_t const&, monomial_indices_t const&);

private:
    data_t data;
    _scalar zero = _scalar(0);

    template <typename _unpacked, typename _callback>
    void for_each(_unpacked& object, int level, monomial_indices_t& indices, _callback callback)
    {
        if constexpr (std::is_same_v<_unpacked, _scalar>)
        {
            callback(object, static_cast<monomial_indices_t const&>(indices));
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
            callback(object, static_cast<monomial_indices_t const&>(indices));
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

    template <int _stop, typename _unpacked, typename _callback_arithmetic>
    void for_each_arithmetic(_unpacked& object, int level, monomial_indices_layered_t& indices, _callback_arithmetic callback)
    {
        if constexpr (_stop <= 0)
        {
            callback(object, static_cast<monomial_indices_layered_t const&>(indices));
        }
        else
        {
            object.for_each
            (
                [&](auto& element, monomial_indices_t const& layer_indices)
                {
                    indices[level] = layer_indices;
                    for_each_arithmetic<_stop - 1>(element, level + 1, indices, callback);
                }
            );
        }
    }

    template <int _stop, typename _unpacked, typename _callback_arithmetic>
    void for_each_arithmetic(_unpacked const& object, int level, monomial_indices_layered_t& indices, _callback_arithmetic callback) const
    {
        if constexpr (_stop <= 0)
        {
            callback(object, static_cast<monomial_indices_layered_t const&>(indices));
        }
        else
        {
            object.for_each
            (
                [&](auto const& element, monomial_indices_t const& layer_indices)
                {
                    indices[level] = layer_indices;
                    for_each_arithmetic<_stop - 1>(element, level + 1, indices, callback);
                }
            );
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

    template <typename _other_scalar>
    polynomial(polynomial<_other_scalar, _n> const& other)
    {
        other.for_each([&](_other_scalar const& element, monomial_indices_t const& indices) { (*this)[indices] = _scalar(element); });
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

    template <int _stop = layers_n, typename _callback_arithmetic = void>
    void for_each_arithmetic(_callback_arithmetic callback)
    {
        monomial_indices_layered_t scratch(_stop);
        for_each_arithmetic<_stop>(*this, 0, scratch, callback);
    }

    template <int _stop = layers_n, typename _callback_arithmetic = void>
    void for_each_arithmetic(_callback_arithmetic callback) const
    {
        monomial_indices_layered_t scratch(_stop);
        for_each_arithmetic<_stop>(*this, 0, scratch, callback);
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

    polynomial_t operator+(_scalar const& other) const
    {
        polynomial_t result = *this;
        return result += other;
    }

    friend polynomial<_scalar, _n> operator+(_scalar const& other, polynomial<_scalar, _n> const& x)
    {
        return x + other;
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

    polynomial_t operator-(_scalar const& other) const
    {
        polynomial_t result = *this;
        return result -= other;
    }

    friend polynomial<_scalar, _n> operator-(_scalar const& other, polynomial<_scalar, _n> const& x)
    {
        return -x + other;
    }

    polynomial_t operator*(polynomial_t const& other) const
    {
        polynomial_t result;
        monomial_indices_t indices_c(_n);

        for_each
        (
            [&](_scalar const& element_a, monomial_indices_t const& indices_a)
            {
                other.for_each
                (
                    [&](_scalar const& element_b, monomial_indices_t const& indices_b)
                    {
                        for (int i = 0; i < _n; ++i) { indices_c[i] = indices_a[i] + indices_b[i]; }
                        result[indices_c] += element_a * element_b;
                    }
                );
            }
        );

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

    polynomial_t operator%(_scalar const& other) const
    {
        polynomial_t result = *this;
        return result %= other;
    }

    polynomial_t& operator+=(polynomial_t const& other)
    {
        other.for_each([&](_scalar const& element, monomial_indices_t const& indices) { (*this)[indices] += element; });
        return *this;
    }

    polynomial_t& operator+=(_scalar const& other)
    {
        (*this)[monomial_indices_t(_n)] += other;
        return *this;
    }

    polynomial_t& operator-=(polynomial_t const& other)
    {
        other.for_each([&](_scalar const& element, monomial_indices_t const& indices) { (*this)[indices] -= element; });
        return *this;
    }

    polynomial_t& operator-=(_scalar const& other)
    {
        (*this)[monomial_indices_t(_n)] -= other;
        return *this;
    }

    polynomial_t& operator*=(polynomial_t const& other)
    {
        *this = *this * other; // true *= has aliasing issues
        return *this;
    }

    polynomial_t& operator*=(_scalar const& other)
    {
        for_each([&](_scalar& element, monomial_indices_t const&) { element *= other; });
        return *this;
    }

    polynomial_t& operator/=(_scalar const& other)
    {
        for_each([&](_scalar& element, monomial_indices_t const&) { element /= other; });
        return *this;
    }

    polynomial_t& operator%=(_scalar const other)
    {
        for_each([&](_scalar& element, monomial_indices_t const&) { element %= other; });
        return *this;
    }
};














template <int _n>
class grevlex_generator
{
public:
    enum { variables_n = _n };

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

template <typename _scalar, int _n, typename A>
polynomial<_scalar, _n> matrix_to_polynomial_grevlex(Eigen::DenseBase<A> const& src)
{
    polynomial<_scalar, _n> dst;
    grevlex_generator<_n> gg;
    for (int i = 0; i < src.size(); ++i) { dst[gg.next()] = src(i); }
    return dst;
}

template <typename _scalar, int _n, int _rows, int _cols, typename A>
Eigen::Matrix<polynomial<_scalar, _n>, _rows, _cols> matrix_to_polynomial_grevlex(Eigen::MatrixBase<A> const& M, int rows = _rows, int cols = _cols)
{
    Eigen::Matrix<polynomial<_scalar, _n>, _rows, _cols> dst(rows, cols);
    for (int i = 0; i < cols; ++i) { for (int j = 0; j < rows; ++j) { dst(j, i) = matrix_to_polynomial_grevlex<_scalar, _n>(M.row((i * rows) + j)); } }
    return dst;
}

template <typename _matrix_scalar, int _rows, int _cols, typename _scalar, int _n>
Eigen::Matrix<_matrix_scalar, _rows, _cols> matrix_from_polynomial_grevlex(polynomial<_scalar, _n> const& src, int rows = _rows, int cols = _cols)
{
    Eigen::Matrix<_matrix_scalar, _rows, _cols> dst(rows, cols);
    auto f = [&](_scalar const& element, monomial_indices_t const& indices)
    {
        int i = grevlex_generator<_n>::ravel(indices);
        if (i < dst.size()) { dst(i) = element; }
    };
    src.for_each(f);
    return dst;
}

template <typename _matrix_scalar, int _rows, int _cols, typename A>
Eigen::Matrix<_matrix_scalar, _rows, _cols> matrix_from_polynomial_grevlex(Eigen::MatrixBase<A> const& src, int rows = _rows, int cols = _cols)
{
    Eigen::Matrix<_matrix_scalar, _rows, _cols> dst(rows, cols);
    Eigen::Index input_rows = src.rows();
    Eigen::Index input_cols = src.cols();
    for (int i = 0; i < input_cols; ++i) { for (int j = 0; j < input_rows; ++j) { dst.row((i * input_rows) + j) = matrix_from_polynomial_grevlex<_matrix_scalar, 1, _cols>(src(j, i), 1, cols); } }
    return dst;
}

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
