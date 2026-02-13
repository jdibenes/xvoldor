
#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <type_traits>

//=============================================================================
// traits
//=============================================================================

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

template <typename T, int n>
using add_vector_type = typename add_vector<T, n>::type;

//=============================================================================
// monomial_indices
//=============================================================================

template <int variables>
using monomial_indices = std::array<int, variables>;

template <int variables>
monomial_indices<variables> operator+(monomial_indices<variables> const& lhs, monomial_indices<variables> const& rhs)
{
    monomial_indices<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = lhs[i] + rhs[i]; }
    return result;
}

template <int variables>
monomial_indices<variables> operator-(monomial_indices<variables> const& lhs, monomial_indices<variables> const& rhs)
{
    monomial_indices<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = lhs[i] - rhs[i]; }
    return result;
}

template <int variables>
bool is_integral(monomial_indices<variables> const& indices)
{
    for (int i = 0; i < variables; ++i) { if (indices[i] < 0) { return false; } };
    return true;
}

template <int variables>
int total_degree(monomial_indices<variables> const& indices)
{
    int sum = 0;
    for (int i = 0; i < variables; ++i) { sum += indices[i]; }
    return sum;
}

//=============================================================================
// monomial
//=============================================================================

template <typename scalar, int variables>
struct monomial
{
    using scalar_type = scalar;
    enum { variables_length = variables };
    using monomial_indices_type = monomial_indices<variables>;

    scalar coefficient;
    monomial_indices_type indices;

    monomial(scalar const& coefficient, monomial_indices_type const& indices) : coefficient{ coefficient }, indices{ indices }
    {
    }
};

template <typename scalar, int variables>
using monomial_vector = std::vector<monomial<scalar, variables>>;

//=============================================================================
// polynomial
//=============================================================================

template <typename scalar, int variables>
class polynomial
{
public:
    //-----------------------------------------------------------------------------
    // type
    //-----------------------------------------------------------------------------

    using scalar_type = scalar;
    enum { variables_length = variables };
    using monomial_indices_type = monomial_indices<variables>;
    using data_type = add_vector_type<scalar, variables>;
    using monomial_type = monomial<scalar, variables>;
    using monomial_vector_type = monomial_vector<scalar, variables>;
    using polynomial_type = polynomial<scalar, variables>;
    using callback_type = bool(scalar&, monomial_indices_type const&);
    using const_callback_type = bool(scalar const&, monomial_indices_type const&);

    template <typename other_scalar>
    using polynomial_other_type = polynomial<other_scalar, variables>;

private:
    //-----------------------------------------------------------------------------
    // data
    //-----------------------------------------------------------------------------

    data_type data;
    scalar zero{ 0 };

    //-----------------------------------------------------------------------------
    // for_each
    //-----------------------------------------------------------------------------

    template <typename unpacked, typename callback_auto>
    void for_each(unpacked& object, int level, monomial_indices_type& indices, callback_auto callback)
    {
        if constexpr (std::is_same_v<unpacked, scalar>)
        {
            if (object != scalar(0)) { callback(object, static_cast<monomial_indices_type const&>(indices)); }
        }
        else
        {
            for (int i = 0; i < object.size(); ++i)
            {
                indices[level] = i;
                for_each(object[i], level + 1, indices, callback);
            }
        }
    }

    template <typename unpacked, typename callback_auto>
    void for_each(unpacked const& object, int level, monomial_indices_type& indices, callback_auto callback) const
    {
        if constexpr (std::is_same_v<unpacked, scalar>)
        {
            if (object != scalar(0)) { callback(object, static_cast<monomial_indices_type const&>(indices)); }
        }
        else
        {
            for (int i = 0; i < object.size(); ++i)
            {
                indices[level] = i;
                for_each(object[i], level + 1, indices, callback);
            }
        }
    }

    //-----------------------------------------------------------------------------
    // at
    //-----------------------------------------------------------------------------

    template <typename unpacked>
    auto& at(unpacked& object, int level, monomial_indices_type const& indices)
    {
        if constexpr (std::is_same_v<unpacked, scalar>)
        {
            return object;
        }
        else
        {
            int index = indices[level];
            if (index >= object.size()) { object.resize(index + 1); }
            return at(object[index], level + 1, indices);
        }
    }

    template <typename unpacked>
    auto const& at(unpacked const& object, int level, monomial_indices_type const& indices) const
    {
        if constexpr (std::is_same_v<unpacked, scalar>)
        {
            return object;
        }
        else
        {
            int index = indices[level];
            if (index >= object.size()) { return zero; }
            return at(object[index], level + 1, indices);
        }
    }

public:
    //-----------------------------------------------------------------------------
    // constructors
    //-----------------------------------------------------------------------------

    polynomial()
    {
    }

    polynomial(scalar const& bias)
    {
        (*this)[{}] = bias;
    }

    polynomial(monomial_type const& m)
    {
        (*this)[m.indices] = m.coefficient;
    }

    polynomial(monomial_vector_type const& v)
    {
        for (auto const& m : v) { (*this)[m.indices] = m.coefficient; }
    }

    template <typename other_scalar>
    polynomial(polynomial_other_type<other_scalar> const& other)
    {
        other.for_each([&](other_scalar const& coefficient, monomial_indices_type const& indices) { (*this)[indices] = scalar(coefficient); });
    }

    //-----------------------------------------------------------------------------
    // access
    //-----------------------------------------------------------------------------

    template <typename callback_auto>
    void for_each(callback_auto callback)
    {
        monomial_indices_type scratch;
        for_each(data, 0, scratch, callback);
    }

    template <typename callback_auto>
    void for_each(callback_auto callback) const
    {
        monomial_indices_type scratch;
        for_each(data, 0, scratch, callback);
    }

    auto& operator[](monomial_indices_type const& indices)
    {
        return at(data, 0, indices);
    }

    auto const& operator[](monomial_indices_type const& indices) const
    {
        return at(data, 0, indices);
    }

    operator monomial_vector_type() const
    {
        monomial_vector_type v;
        for_each([&](scalar const& coefficient, monomial_indices_type const& indices) { v.push_back({ coefficient, indices }); });
        return v;
    }

    //-----------------------------------------------------------------------------
    // +
    //-----------------------------------------------------------------------------

    polynomial_type operator+() const
    {
        return *this;
    }

    polynomial_type operator+(polynomial_type const& other) const
    {
        polynomial_type result = *this;
        return result += other;
    }

    polynomial_type operator+(monomial_vector_type const& other) const
    {
        polynomial_type result = *this;
        return result += other;
    }

    polynomial_type operator+(monomial_type const& other) const
    {
        polynomial_type result = *this;
        return result += other;
    }

    polynomial_type operator+(scalar const& other) const
    {
        polynomial_type result = *this;
        return result += other;
    }

    friend polynomial_type operator+(monomial_vector_type const& other, polynomial_type const& x)
    {
        return x + other;
    }

    friend polynomial_type operator+(monomial_type const& other, polynomial_type const& x)
    {
        return x + other;
    }

    friend polynomial_type operator+(scalar const& other, polynomial_type const& x)
    {
        return x + other;
    }

    //-----------------------------------------------------------------------------
    // +=
    //-----------------------------------------------------------------------------

    polynomial_type& operator+=(polynomial_type const& other)
    {
        other.for_each([&](scalar const& coefficient, monomial_indices_type const& indices) { (*this)[indices] += coefficient; });
        return *this;
    }

    polynomial_type& operator+=(monomial_vector_type const& other)
    {
        for (auto const& m : other) { (*this)[m.indices] += m.coefficient; }
        return *this;
    }

    polynomial_type& operator+=(monomial_type const& other)
    {
        (*this)[other.indices] += other.coefficient;
        return *this;
    }

    polynomial_type& operator+=(scalar const& other)
    {
        (*this)[{}] += other;
        return *this;
    }

    //-----------------------------------------------------------------------------
    // -
    //-----------------------------------------------------------------------------

    polynomial_type operator-() const
    {
        polynomial_type result;
        return result -= *this;
    }

    polynomial_type operator-(polynomial_type const& other) const
    {
        polynomial_type result = *this;
        return result -= other;
    }

    polynomial_type operator-(monomial_vector_type const& other) const
    {
        polynomial_type result = *this;
        return result -= other;
    }

    polynomial_type operator-(monomial_type const& other) const
    {
        polynomial_type result = *this;
        return result -= other;
    }

    polynomial_type operator-(scalar const& other) const
    {
        polynomial_type result = *this;
        return result -= other;
    }

    friend polynomial_type operator-(monomial_vector_type const& other, polynomial_type const& x)
    {
        return -x + other;
    }

    friend polynomial_type operator-(monomial_type const& other, polynomial_type const& x)
    {
        return -x + other;
    }

    friend polynomial_type operator-(scalar const& other, polynomial_type const& x)
    {
        return -x + other;
    }

    //-----------------------------------------------------------------------------
    // -=
    //-----------------------------------------------------------------------------

    polynomial_type& operator-=(polynomial_type const& other)
    {
        other.for_each([&](scalar const& coefficient, monomial_indices_type const& indices) { (*this)[indices] -= coefficient; });
        return *this;
    }

    polynomial_type& operator-=(monomial_vector_type const& other)
    {
        for (auto const& m : other) { (*this)[m.indices] -= m.coefficient; }
        return *this;
    }

    polynomial_type& operator-=(monomial_type const& other)
    {
        (*this)[other.indices] -= other.coefficient;
        return *this;
    }

    polynomial_type& operator-=(scalar const& other)
    {
        (*this)[{}] -= other;
        return *this;
    }

    //-----------------------------------------------------------------------------
    // *
    //-----------------------------------------------------------------------------

    polynomial_type operator*(polynomial_type const& other) const
    {
        polynomial_type result;
        for_each([&](scalar const& coefficient_a, monomial_indices_type const& indices_a) { other.for_each([&](scalar const& coefficient_b, monomial_indices_type const& indices_b) { result[indices_a + indices_b] += coefficient_a * coefficient_b; }); });
        return result;
    }

    polynomial_type operator*(monomial_vector_type const& other) const
    {
        polynomial_type result;
        for_each([&](scalar const& coefficient, monomial_indices_type const& indices) { for (auto const& m : other) { result[indices + m.indices] += coefficient * m.coefficient; } });
        return result;
    }

    polynomial_type operator*(monomial_type const& other) const
    {
        polynomial_type result;
        for_each([&](scalar const& coefficient, monomial_indices_type const& indices) { result[indices + other.indices] += coefficient * other.coefficient; });
        return result;
    }

    polynomial_type operator*(scalar const& other) const
    {
        polynomial_type result = *this;
        return result *= other;
    }

    friend polynomial_type operator*(monomial_vector_type const& other, polynomial_type const& x)
    {
        return x * other;
    }

    friend polynomial_type operator*(monomial_type const& other, polynomial_type const& x)
    {
        return x * other;
    }

    friend polynomial_type operator*(scalar const& other, polynomial_type const& x)
    {
        return x * other;
    }

    //-----------------------------------------------------------------------------
    // *=
    //-----------------------------------------------------------------------------

    polynomial_type& operator*=(polynomial_type const& other)
    {
        *this = *this * other; // true *= has aliasing issues
        return *this;
    }

    polynomial_type& operator*=(monomial_vector_type const& other)
    {
        *this = *this * other; // true *= has aliasing issues
        return *this;
    }

    polynomial_type& operator*=(monomial_type const& other)
    {
        *this = *this * other; // true *= has aliasing issues
        return *this;
    }

    polynomial_type& operator*=(scalar const& other)
    {
        for_each([&](scalar& coefficient, monomial_indices_type const& indices) { coefficient *= other; });
        return *this;
    }

    //-----------------------------------------------------------------------------
    // /
    //-----------------------------------------------------------------------------

    // DIVISION BY COEFFICIENT OF '1' NOT POLYNOMIAL DIVISION
    polynomial_type operator/(polynomial_type const& other) const
    {
        polynomial result = *this;
        return result /= other[{}];
    }

    polynomial_type operator/(scalar const& other) const
    {
        polynomial result = *this;
        return result /= other;
    }

    //-----------------------------------------------------------------------------
    // /=
    //-----------------------------------------------------------------------------

    // DIVISION BY COEFFICIENT OF '1' NOT POLYNOMIAL DIVISION
    polynomial_type& operator/=(polynomial_type const& other)
    {
        *this /= other[{}];
        return *this;
    }

    polynomial_type& operator/=(scalar const& other)
    {
        for_each([&](scalar& coefficient, monomial_indices_type const& indices) { coefficient /= other; });
        return *this;
    }

    //-----------------------------------------------------------------------------
    // %
    //-----------------------------------------------------------------------------

    // DIVISION BY COEFFICIENT OF '1' NOT POLYNOMIAL DIVISION
    polynomial_type operator%(polynomial_type const& other) const
    {
        polynomial result = *this;
        return result %= other[{}];
    }

    polynomial_type operator%(scalar const& other) const
    {
        polynomial_type result = *this;
        return result %= other;
    }

    //-----------------------------------------------------------------------------
    // %=
    //-----------------------------------------------------------------------------

    // DIVISION BY COEFFICIENT OF '1' NOT POLYNOMIAL DIVISION
    polynomial_type& operator%=(polynomial_type const& other)
    {
        *this %= other[{}];
        return *this;
    }

    polynomial_type& operator%=(scalar const& other)
    {
        for_each([&](scalar& coefficient, monomial_indices_type const& indices) { coefficient %= other; });
        return *this;
    }

    //-----------------------------------------------------------------------------
    // compare
    //-----------------------------------------------------------------------------

    explicit operator bool() const
    {
        bool set = false;
        for_each([&](scalar const& coefficient, monomial_indices_type const& indices) { set = true; });
        return set;
    }

    bool operator!() const
    {
        return !(*this);
    }

    bool operator==(polynomial_type const& other) const
    {
        bool set = true;
        for_each([&](scalar const& coefficient, monomial_indices_type const& indices) { set = set && (other[indices] == coefficient); });
        other.for_each([&](scalar const& coefficient, monomial_indices_type const& indices) { set = set && ((*this)[indices] == coefficient); });
        return set;
    }

    bool operator!=(polynomial_type const& other) const
    {
        return !((*this) == other);
    }
};

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

//=============================================================================
// grevlex_generator
//=============================================================================

template <int variables>
class grevlex_generator
{
public:
    enum { variables_length = variables};
    using monomial_indices_type = monomial_indices<variables>;

private:
    monomial_indices_type indices;
    int power;
    int sum;
    int index;

    grevlex_generator(monomial_indices_type const& start_indices, int start_index)
    {
        indices = start_indices;
        power = total_degree(indices);
        sum = power;        
        index = start_index;
    }

public:
    grevlex_generator() : indices{}, power{ -1 }, sum{ 0 }, index{ -1 }
    {
    }

    grevlex_generator(int start_index) : grevlex_generator(unravel(start_index), start_index)
    {
    }

    grevlex_generator(monomial_indices_type const& start_indices) : grevlex_generator(start_indices, ravel(start_indices))
    {
    }

    grevlex_generator& next()
    {
        do
        {
            if (sum > 0)
            {
                int i;
                for (i = variables - 1; (i >= 0) && (indices[i] <= 0); --i)
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
        return *this;
    }

    grevlex_generator& previous()
    {
        if (power <= 0) { return indices; }
        do
        {
            if (indices[0] < power)
            {
                int i;
                for (i = variables - 1; (i >= 0) && (indices[i] >= power); --i)
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
        return *this;
    }

    monomial_indices_type const& current_indices() const
    {
        return indices;
    }

    int current_power() const
    {
        return power;
    }

    int current_index() const
    {
        return index;
    }

    static bool compare(monomial_indices_type const& a, monomial_indices_type const& b)
    {
        int ap = total_degree(a);
        int bp = total_degree(b);

        if (ap != bp) { return ap < bp; }
        for (int i = 0; i < variables; ++i) { if (a[i] > b[i]) { return true; } }
        return false;
    }

    static monomial_indices_type unravel(int index)
    {
        grevlex_generator<variables> gg;
        for (int i = 0; i < index; ++i) { gg.next(); }
        return gg.next().current_indices();
    }

    static int ravel(monomial_indices_type const& indices)
    {
        grevlex_generator<variables> gg;
        while (indices != gg.next().current_indices());
        return gg.current_index();
    }
};

//=============================================================================
// conversions
//=============================================================================











template <typename _scalar, int variables, typename _iterable>
polynomial<_scalar, variables> create_polynomial_grevlex(_iterable const& coefficients)
{
    grevlex_generator<variables> gg;
    polynomial<_scalar, variables> p;
    for (auto const& c : coefficients) { p[gg.next().current_indices()] = c; }
    return p;
}

template <typename _scalar, int variables>
polynomial<_scalar, variables> create_polynomial_grevlex(std::initializer_list<_scalar> coefficients)
{
    grevlex_generator<variables> gg;
    polynomial<_scalar, variables> p;
    for (auto const& c : coefficients) { p[gg.next().current_indices()] = c; }
    return p;
}




















template <typename _scalar, int variables, typename A>
polynomial<_scalar, variables> matrix_to_polynomial_grevlex(Eigen::DenseBase<A> const& src)
{
    polynomial<_scalar, variables> dst;
    grevlex_generator<variables> gg;
    for (int i = 0; i < src.size(); ++i) { dst[gg.next().current_indices()] = src(i); }
    return dst;
}

template <typename _scalar, int variables, int _rows, int _cols, typename A>
Eigen::Matrix<polynomial<_scalar, variables>, _rows, _cols> matrix_to_polynomial_grevlex(Eigen::MatrixBase<A> const& M, int rows = _rows, int cols = _cols)
{
    Eigen::Matrix<polynomial<_scalar, variables>, _rows, _cols> dst(rows, cols);
    for (int i = 0; i < cols; ++i) { for (int j = 0; j < rows; ++j) { dst(j, i) = matrix_to_polynomial_grevlex<_scalar, variables>(M.row((i * rows) + j)); } }
    return dst;
}

template <typename _matrix_scalar, int _rows, int _cols, typename _scalar, int variables>
Eigen::Matrix<_matrix_scalar, _rows, _cols> matrix_from_polynomial_grevlex(polynomial<_scalar, variables> const& src, int rows = _rows, int cols = _cols)
{
    Eigen::Matrix<_matrix_scalar, _rows, _cols> dst = Eigen::Matrix<_matrix_scalar, _rows, _cols>::Zero(rows, cols); // ZEROS!??? CHECK INITIALIZATIONS
    src.for_each
    (
        [&](_scalar const& element, monomial_indices<variables> const& indices)
        {
            int i = grevlex_generator<variables>::ravel(indices);
            if (i < dst.size()) { dst(i) = element; }
        }
    );
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
