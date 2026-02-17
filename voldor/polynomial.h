
#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <sstream>

//=============================================================================
// add_vector
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

//-----------------------------------------------------------------------------
// +
//-----------------------------------------------------------------------------

template <int variables>
monomial_indices<variables> operator+(monomial_indices<variables> const& x)
{
    return x;
}

template <int variables>
monomial_indices<variables> operator+(monomial_indices<variables> const& lhs, monomial_indices<variables> const& rhs)
{
    monomial_indices<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = lhs[i] + rhs[i]; }
    return result;
}

//-----------------------------------------------------------------------------
// +=
//-----------------------------------------------------------------------------

template <int variables>
monomial_indices<variables>& operator+=(monomial_indices<variables>& lhs, monomial_indices<variables> const& rhs)
{
    for (int i = 0; i < variables; ++i) { lhs[i] += rhs[i]; }
    return lhs;
}

//-----------------------------------------------------------------------------
// -
//-----------------------------------------------------------------------------

template <int variables>
monomial_indices<variables> operator-(monomial_indices<variables> const& x)
{
    monomial_indices<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = -x[i]; }
    return result;
}

template <int variables>
monomial_indices<variables> operator-(monomial_indices<variables> const& lhs, monomial_indices<variables> const& rhs)
{
    monomial_indices<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = lhs[i] - rhs[i]; }
    return result;
}

//-----------------------------------------------------------------------------
// -=
//-----------------------------------------------------------------------------

template <int variables>
monomial_indices<variables>& operator-=(monomial_indices<variables>& lhs, monomial_indices<variables> const& rhs)
{
    for (int i = 0; i < variables; ++i) { lhs[i] -= rhs[i]; }
    return lhs;
}

//-----------------------------------------------------------------------------
// <<
//-----------------------------------------------------------------------------

template <int variables>
std::ostream& operator<<(std::ostream& os, monomial_indices<variables> const& indices)
{
    bool first = true;
    for (int i = 0; i < variables; ++i)
    {
        if (indices[i] != 0)
        {
            if (!first) { os << "*"; }
            first = false;
            os << "x_" << i;
            if (indices[i] != 1) { os << "^" << indices[i]; }
        }
    }
    return os;
}

//-----------------------------------------------------------------------------
// compare
//-----------------------------------------------------------------------------

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

template <int variables>
monomial_indices<variables> gcd(monomial_indices<variables> const& lhs, monomial_indices<variables> const& rhs)
{
    monomial_indices<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = std::min(lhs[i], rhs[i]); }
    return result;
}

template <int variables>
monomial_indices<variables> lcm(monomial_indices<variables> const& lhs, monomial_indices<variables> const& rhs)
{
    monomial_indices<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = std::max(lhs[i], rhs[i]); }
    return result;
}

template <int variables>
monomial_indices<variables - 1> split(monomial_indices<variables> const& indices, int index)
{
    monomial_indices<variables - 1> result;
    int j = 0;
    for (int i = 0; i < variables; ++i) { if (i != index) { result[j++] = indices[i]; } }
    return result;
}

template <int variables>
monomial_indices<variables + 1> merge(monomial_indices<variables> const& indices, int index, monomial_indices<1> const& pick)
{
    monomial_indices<variables + 1> result;
    int j = 0;
    for (int i = 0; i < (variables + 1); ++i) { if (i != index) { result[i] = indices[j++]; } else { result[i] = pick[0]; } }
    return result;
}

//=============================================================================
// monomial_mask
//=============================================================================

template <int variables>
using monomial_mask = std::array<bool, variables>;

//-----------------------------------------------------------------------------
// ~
//-----------------------------------------------------------------------------

template <int variables>
monomial_mask<variables> operator~(monomial_mask<variables> const& x)
{
    monomial_mask<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = !x[i]; }
    return result;
}

//-----------------------------------------------------------------------------
// |
//-----------------------------------------------------------------------------

template <int variables>
monomial_mask<variables> operator|(monomial_mask<variables> const& lhs, monomial_mask<variables> const& rhs)
{
    monomial_mask<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = lhs[i] || rhs[i]; }
    return result;
}

//-----------------------------------------------------------------------------
// |=
//-----------------------------------------------------------------------------

template <int variables>
monomial_mask<variables>& operator|=(monomial_mask<variables>& lhs, monomial_mask<variables> const& rhs)
{
    for (int i = 0; i < variables; ++i) { lhs[i] = lhs[i] || rhs[i]; }
    return lhs;
}

//-----------------------------------------------------------------------------
// &
//-----------------------------------------------------------------------------

template <int variables>
monomial_mask<variables> operator&(monomial_mask<variables> const& lhs, monomial_mask<variables> const& rhs)
{
    monomial_mask<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = lhs[i] && rhs[i]; }
    return result;
}

//-----------------------------------------------------------------------------
// &=
//-----------------------------------------------------------------------------

template <int variables>
monomial_mask<variables>& operator&=(monomial_mask<variables>& lhs, monomial_mask<variables> const& rhs)
{
    for (int i = 0; i < variables; ++i) { lhs[i] = lhs[i] && rhs[i]; }
    return lhs;
}

//-----------------------------------------------------------------------------
// ^
//-----------------------------------------------------------------------------

template <int variables>
monomial_mask<variables> operator^(monomial_mask<variables> const& lhs, monomial_mask<variables> const& rhs)
{
    monomial_mask<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = (!lhs[i] && rhs[i]) || (lhs[i] && !rhs[i]); }
    return result;
}

//-----------------------------------------------------------------------------
// ^=
//-----------------------------------------------------------------------------

template <int variables>
monomial_mask<variables>& operator^=(monomial_mask<variables>& lhs, monomial_mask<variables> const& rhs)
{
    for (int i = 0; i < variables; ++i) { lhs[i] = (!lhs[i] && rhs[i]) || (lhs[i] && !rhs[i]); }
    return lhs;
}

//-----------------------------------------------------------------------------
// compare
//-----------------------------------------------------------------------------

template <int variables>
monomial_indices<variables> select(monomial_indices<variables> const indices, monomial_mask<variables> const& mask, bool complement)
{
    monomial_indices<variables> result;
    for (int i = 0; i < variables; ++i) { result[i] = (mask[i] - complement) ? indices[i] : 0; }
    return result;
}

//=============================================================================
// monomial_values
//=============================================================================

template <typename scalar, int variables>
using monomial_values = std::array<scalar, variables>;

template <typename scalar, int variables>
monomial_values<scalar, variables + 1> merge(monomial_values<scalar, variables> const& indices, int index, monomial_values<scalar, 1> const& pick)
{
    monomial_values<scalar, variables + 1> result;
    int j = 0;
    for (int i = 0; i < (variables + 1); ++i) { if (i != index) { result[i] = indices[j++]; } else { result[i] = pick[0]; } }
    return result;
}

//=============================================================================
// monomial_powers
//=============================================================================

template <typename scalar, int variables>
class monomial_powers
{
public:
    //-------------------------------------------------------------------------
    // type
    //-------------------------------------------------------------------------

    using scalar_type = scalar;
    enum { variables_length = variables };
    using monomial_indices_type = monomial_indices<variables>;
    using powers_type = std::array<std::vector<scalar>, variables>;
    using values_type = std::array<scalar, variables>;
    using cached_type = add_vector_type<std::tuple<scalar, bool>, variables>;
    using monomial_powers_type = monomial_powers<scalar, variables>;

private:
    //-------------------------------------------------------------------------
    // data
    //-------------------------------------------------------------------------

    powers_type powers;
    values_type values;
    cached_type cached;

    //-------------------------------------------------------------------------
    // at
    //-------------------------------------------------------------------------

    template <int level, typename unpacked>
    auto& at(unpacked& object, monomial_indices_type const& indices)
    {
        if constexpr (level >= variables)
        {
            return object;
        }
        else
        {
            if (indices[level] >= object.size())
            {
                if constexpr (level == (variables - 1))
                {
                    object.resize(indices[level] + 1, std::make_tuple<scalar, bool>({ 1 }, false));
                }
                else
                {
                    object.resize(indices[level] + 1);
                }
            }
            return at<level + 1>(object[indices[level]], indices);
        }
    }

public:
    //-------------------------------------------------------------------------
    // constructors
    //-------------------------------------------------------------------------

    monomial_powers(values_type const& values) : values{ values }
    {
        for (int i = 0; i < variables; ++i) { powers[i].push_back({ 1 }); }
    }

    //-------------------------------------------------------------------------
    // access
    //-------------------------------------------------------------------------

    auto const& operator[](monomial_indices_type const& indices)
    {
        std::tuple<scalar, bool>& stored = at<0>(cached, indices);
        scalar& value = std::get<0>(stored);
        bool& valid = std::get<1>(stored);
        if (valid) { return value; }
        for (int i = 0; i < variables; ++i)
        {
            if (indices[i] >= powers[i].size()) { for (int j = static_cast<int>(powers[i].size()) - 1; j < indices[i]; ++j) { powers[i].push_back(powers[i][j] * values[i]); } }
            if (indices[i] > 0) { value *= powers[i][indices[i]]; }
        }
        valid = true;
        return value;
    }
};

//=============================================================================
// monomial
//=============================================================================

template <typename scalar, int variables>
class monomial
{
public:
    //-------------------------------------------------------------------------
    // type
    //-------------------------------------------------------------------------

    using scalar_type = scalar;
    enum { variables_length = variables };
    using monomial_indices_type = monomial_indices<variables>;
    using monomial_type = monomial<scalar, variables>;

    //-------------------------------------------------------------------------
    // data
    //-------------------------------------------------------------------------

    scalar coefficient;
    monomial_indices_type indices;

    //-------------------------------------------------------------------------
    // constructors
    //-------------------------------------------------------------------------

    monomial() : coefficient{ 0 }, indices{}
    {
    }

    monomial(scalar const& coefficient) : coefficient{ coefficient }, indices{}
    {
    }

    monomial(scalar const& coefficient, monomial_indices_type const& indices) : coefficient{ coefficient }, indices{ indices }
    {
    }

    //-------------------------------------------------------------------------
    // +
    //-------------------------------------------------------------------------

    monomial_type operator+() const
    {
        return (*this);
    }

    //-------------------------------------------------------------------------
    // -
    //-------------------------------------------------------------------------

    monomial_type operator-() const
    {
        return monomial_type(-coefficient, indices);
    }

    //-------------------------------------------------------------------------
    // *
    //-------------------------------------------------------------------------

    monomial_type operator*(monomial_type const& other) const
    {
        return monomial_type(coefficient * other.coefficient, indices + other.indices);
    }

    monomial_type operator*(scalar const& other) const
    {
        return monomial_type(coefficient * other, indices);
    }

    friend monomial_type operator*(scalar const& other, monomial_type const& x)
    {
        return x * other;
    }

    //-------------------------------------------------------------------------
    // *=
    //-------------------------------------------------------------------------

    monomial_type& operator*=(monomial_type const& other)
    {
        coefficient *= other.coefficient;
        indices += other.indices;
        return *this;
    }

    monomial_type& operator*=(scalar const& other)
    {
        coefficient *= other;
        return *this;
    }

    //-------------------------------------------------------------------------
    // /
    //-------------------------------------------------------------------------

    monomial_type operator/(scalar const& other) const
    {
        return monomial_type(coefficient / other, indices);
    }

    //-------------------------------------------------------------------------
    // /=
    //-------------------------------------------------------------------------

    monomial_type& operator/=(scalar const& other)
    {
        coefficient /= other;
        return *this;
    }

    //-------------------------------------------------------------------------
    // %
    //-------------------------------------------------------------------------

    monomial_type operator%(scalar const& other) const
    {
        return monomial_type(coefficient % other, indices);
    }

    //-------------------------------------------------------------------------
    // %=
    //-------------------------------------------------------------------------

    monomial_type& operator%=(scalar const& other)
    {
        coefficient %= other;
        return *this;
    }

    //-------------------------------------------------------------------------
    // <<
    //-------------------------------------------------------------------------

    friend std::ostream& operator<<(std::ostream& os, monomial_type const& x)
    {
        os << "(" << x.coefficient << ")";
        if (x.indices != monomial_indices_type{}) { os << "*" << x.indices; }
        return os;
    }

    //-------------------------------------------------------------------------
    // compare
    //-------------------------------------------------------------------------

    explicit operator bool() const
    {
        return coefficient;
    }

    bool operator!() const
    {
        return !coefficient;
    }

    bool operator==(monomial_type const& other) const
    {
        return (!coefficient && !other.coefficient) || ((coefficient == other.coefficient) && (indices == other.indices));
    }

    bool operator!=(monomial_type const& other) const
    {
        return !((*this) == other);
    }

    bool is_constant() const
    {
        return !coefficient || (indices == monomial_indices_type{});
    }

    bool is_homogeneous() const
    {
        return !coefficient || (indices != monomial_indices_type{});
    }

    bool is_multiple(monomial_indices_type const& f) const
    {
        return is_integral(indices - f);
    }
};

//=============================================================================
// monomial_vector
//=============================================================================

template <typename scalar, int variables>
using monomial_vector = std::vector<monomial<scalar, variables>>;

//-----------------------------------------------------------------------------
// +
//-----------------------------------------------------------------------------

template <typename scalar, int variables>
monomial_vector<scalar, variables> operator+(monomial_vector<scalar, variables> const& x)
{
    return x;
}

//-----------------------------------------------------------------------------
// -
//-----------------------------------------------------------------------------

template <typename scalar, int variables>
monomial_vector<scalar, variables> operator-(monomial_vector<scalar, variables> const& x)
{
    monomial_vector<scalar, variables> result = x;
    for (auto& m : result) { m = -m; }
    return x;
}

//-----------------------------------------------------------------------------
// *=
//-----------------------------------------------------------------------------

template <typename scalar, int variables>
monomial_vector<scalar, variables>& operator*=(monomial_vector<scalar, variables>& lhs, monomial<scalar, variables> const& rhs)
{
    for (auto& m : lhs) { m *= rhs; }
    return lhs;
}

template <typename scalar, int variables>
monomial_vector<scalar, variables>& operator*=(monomial_vector<scalar, variables>& lhs, scalar const& rhs)
{
    for (auto& m : lhs) { m *= rhs; }
    return lhs;
}

//-----------------------------------------------------------------------------
// *
//-----------------------------------------------------------------------------

template <typename scalar, int variables>
monomial_vector<scalar, variables> operator*(monomial_vector<scalar, variables> const& lhs, monomial<scalar, variables> const& rhs)
{
    monomial_vector<scalar, variables> result = lhs;
    return result *= rhs;
}

template <typename scalar, int variables>
monomial_vector<scalar, variables> operator*(monomial<scalar, variables> const& lhs, monomial_vector<scalar, variables> const& rhs)
{
    return rhs * lhs;
}

template <typename scalar, int variables>
monomial_vector<scalar, variables> operator*(monomial_vector<scalar, variables> const& lhs, scalar const& rhs)
{
    monomial_vector<scalar, variables> result = lhs;
    return result *= rhs;
}

template <typename scalar, int variables>
monomial_vector<scalar, variables> operator*(scalar const& lhs, monomial_vector<scalar, variables> const& rhs)
{
    return rhs * lhs;
}

//-----------------------------------------------------------------------------
// /=
//-----------------------------------------------------------------------------

template <typename scalar, int variables>
monomial_vector<scalar, variables>& operator/=(monomial_vector<scalar, variables>& lhs, scalar const& rhs)
{
    for (auto& m : lhs) { m /= rhs; }
    return lhs;
}

//-----------------------------------------------------------------------------
// /
//-----------------------------------------------------------------------------

template <typename scalar, int variables>
monomial_vector<scalar, variables> operator/(monomial_vector<scalar, variables> const& lhs, scalar const& rhs)
{
    monomial_vector<scalar, variables> result = lhs;
    return result /= rhs;
}

//-----------------------------------------------------------------------------
// %=
//-----------------------------------------------------------------------------

template <typename scalar, int variables>
monomial_vector<scalar, variables>& operator%=(monomial_vector<scalar, variables>& lhs, scalar const& rhs)
{
    for (auto& m : lhs) { m %= rhs; }
    return lhs;
}

//-----------------------------------------------------------------------------
// %
//-----------------------------------------------------------------------------

template <typename scalar, int variables>
monomial_vector<scalar, variables> operator%(monomial_vector<scalar, variables> const& lhs, scalar const& rhs)
{
    monomial_vector<scalar, variables> result = lhs;
    return result %= rhs;
}

//-----------------------------------------------------------------------------
// <<
//-----------------------------------------------------------------------------

template <typename scalar, int variables>
std::ostream& operator<<(std::ostream& os, monomial_vector<scalar, variables> const& v)
{
    os << "[";
    bool first = true;
    for (auto const& m : v)
    {
        if (!first) { os << " , "; }
        first = false;
        os << m;
    }
    os << "]";
    return os;
}

//-----------------------------------------------------------------------------
// compare
//-----------------------------------------------------------------------------

template <typename scalar, int variables>
monomial_vector<scalar, variables> find_multiples(monomial_vector<scalar, variables> const& v, monomial_indices<variables> const& f)
{
    monomial_vector<scalar, variables> multiples;
    for (auto const& m : v) { if (m.is_multiple(f)) { multiples.push_back(m); } }
    return multiples;
}

//=============================================================================
// polynomial
//=============================================================================

template <typename scalar, int variables>
class polynomial
{
public:
    //-------------------------------------------------------------------------
    // type
    //-------------------------------------------------------------------------

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
    //-------------------------------------------------------------------------
    // data
    //-------------------------------------------------------------------------

    data_type data;
    scalar zero{ 0 };

    //-------------------------------------------------------------------------
    // for_each
    //-------------------------------------------------------------------------

    template <int level, typename unpacked, typename callback_auto>
    void for_each(unpacked& object, monomial_indices_type& indices, callback_auto callback)
    {
        if constexpr (level >= variables)
        {
            if (object) { callback(object, static_cast<monomial_indices_type const&>(indices)); }
        }
        else
        {
            for (indices[level] = 0; indices[level] < object.size(); ++indices[level]) { for_each<level + 1>(object[indices[level]], indices, callback); }
        }
    }

    template <int level, typename unpacked, typename callback_auto>
    void for_each(unpacked const& object, monomial_indices_type& indices, callback_auto callback) const
    {
        if constexpr (level >= variables)
        {
            if (object) { callback(object, static_cast<monomial_indices_type const&>(indices)); }
        }
        else
        {
            for (indices[level] = 0; indices[level] < object.size(); ++indices[level]) { for_each<level + 1>(object[indices[level]], indices, callback); }
        }
    }

    //-------------------------------------------------------------------------
    // do_while
    //-------------------------------------------------------------------------

    template <int level, typename unpacked, typename callback_auto>
    bool do_while(unpacked& object, monomial_indices_type& indices, callback_auto callback)
    {
        if constexpr (level >= variables)
        {
            if (object) { return callback(object, static_cast<monomial_indices_type const&>(indices)); }
        }
        else
        {
            for (indices[level] = 0; indices[level] < object.size(); ++indices[level]) { if (!do_while<level + 1>(object[indices[level]], indices, callback)) { return false; } }
        }
        return true;
    }

    template <int level, typename unpacked, typename callback_auto>
    bool do_while(unpacked const& object, monomial_indices_type& indices, callback_auto callback) const
    {
        if constexpr (level >= variables)
        {
            if (object) { return callback(object, static_cast<monomial_indices_type const&>(indices)); }
        }
        else
        {
            for (indices[level] = 0; indices[level] < object.size(); ++indices[level]) { if (!do_while<level + 1>(object[indices[level]], indices, callback)) { return false; } }
        }
        return true;
    }

    //-------------------------------------------------------------------------
    // at
    //-------------------------------------------------------------------------

    template <int level, typename unpacked>
    auto& at(unpacked& object, monomial_indices_type const& indices)
    {
        if constexpr (level >= variables)
        {
            return object;
        }
        else
        {
            if (indices[level] >= object.size())
            {
                if constexpr (level == (variables - 1))
                {
                    object.resize(indices[level] + 1, zero); 
                }
                else
                {
                    object.resize(indices[level] + 1);
                }
            }
            return at<level + 1>(object[indices[level]], indices);
        }
    }

    template <int level, typename unpacked>
    auto const& at(unpacked const& object, monomial_indices_type const& indices) const
    {
        if constexpr (level >= variables)
        {
            return object;
        }
        else
        {
            if (indices[level] >= object.size()) { return zero; }
            return at<level + 1>(object[indices[level]], indices);
        }
    }

public:
    //-------------------------------------------------------------------------
    // constructors
    //-------------------------------------------------------------------------

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
        for (auto const& m : v) { (*this)[m.indices] += m.coefficient; }
    }

    //template <typename other_scalar>
    //polynomial(polynomial_other_type<other_scalar> const& other)
    //{
    //    other.for_each([&](other_scalar const& coefficient, monomial_indices_type const& indices) { (*this)[indices] = scalar(coefficient); });
    //}

    //-------------------------------------------------------------------------
    // access
    //-------------------------------------------------------------------------

    template <typename callback_auto>
    void for_each(callback_auto callback)
    {
        monomial_indices_type scratch;
        for_each<0>(data, scratch, callback);
    }

    template <typename callback_auto>
    void for_each(callback_auto callback) const
    {
        monomial_indices_type scratch;
        for_each<0>(data, scratch, callback);
    }

    template <typename callback_auto>
    bool do_while(callback_auto callback)
    {
        monomial_indices_type scratch;
        return do_while<0>(data, scratch, callback);
    }

    template <typename callback_auto>
    bool do_while(callback_auto callback) const
    {
        monomial_indices_type scratch;
        return do_while<0>(data, scratch, callback);
    }

    auto& operator[](monomial_indices_type const& indices)
    {
        return at<0>(data, indices);
    }

    auto const& operator[](monomial_indices_type const& indices) const
    {
        return at<0>(data, indices);
    }

    operator monomial_vector_type() const
    {
        monomial_vector_type v;
        for_each([&](scalar const& coefficient, monomial_indices_type const& indices) { v.push_back({ coefficient, indices }); });
        return v;
    }

    //-------------------------------------------------------------------------
    // +
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // +=
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // -
    //-------------------------------------------------------------------------

    polynomial_type operator-() const
    {
        polynomial_type result;
        for_each([&](scalar const& coefficient, monomial_indices_type const& indices) { result[indices] = -coefficient; });
        return result;
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

    //-------------------------------------------------------------------------
    // -=
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // *
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // *=
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // /
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // /=
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // %
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // %=
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // <<
    //-------------------------------------------------------------------------

    friend std::ostream& operator<<(std::ostream& os, polynomial_type const& x)
    {
        bool first = true;
        x.for_each
        (
            [&](scalar const& coefficient, monomial_indices_type const& indices)
            {
                if (!first) { os << " + "; }
                first = false;
                os << "(" << coefficient << ")";
                if (indices != monomial_indices_type{}) { os << "*" << indices; }                
            }
        );
        if (first) { os << "(0)"; }
        return os;
    }

    //-------------------------------------------------------------------------
    // compare
    //-------------------------------------------------------------------------

    explicit operator bool() const
    {
        return !do_while([&](scalar const& coefficient, monomial_indices_type const& indices) { return false; });
    }

    bool operator!() const
    {
        return !(*this);
    }

    bool operator==(polynomial_type const& other) const
    {
        return do_while([&](scalar const& coefficient, monomial_indices_type const& indices) { return other[indices] == coefficient; }) && other.do_while([&](scalar const& coefficient, monomial_indices_type const& indices) { return (*this)[indices] == coefficient; });
    }

    bool operator!=(polynomial_type const& other) const
    {
        return !((*this) == other);
    }

    bool is_constant() const
    {
        return do_while([&](scalar const& coefficient, monomial_indices_type const& indices) { return indices == monomial_indices_type{}; });
    }

    bool is_homogeneous() const
    {
        return !(*this)[{}];
    }
};

//-----------------------------------------------------------------------------
// process
//-----------------------------------------------------------------------------

template <typename scalar, int variables>
monomial_vector<scalar, variables> find_multiples(polynomial<scalar, variables> const& p, monomial_indices<variables> const& f)
{
    monomial_vector<scalar, variables> multiples;
    p.for_each([&](scalar const& coefficient, monomial_indices<variables> const& indices) { if (is_integral(indices - f)) { multiples.push_back({ coefficient, indices }); } });
    return multiples;
}

template <typename scalar, int variables>
polynomial<polynomial<scalar, 1>, variables - 1> hide_in(polynomial<scalar, variables> const& p, int index)
{
    polynomial<polynomial<scalar, 1>, variables - 1> result;
    p.for_each([&](scalar const& coefficent, monomial_indices<variables> const& indices) { result[split(indices, index)][{ indices[index] }] += coefficent; });
    return result;
}

template <typename scalar, int variables>
polynomial<polynomial<scalar, variables - 1>, 1> hide_out(polynomial<scalar, variables> const& p, int index)
{
    polynomial<polynomial<scalar, variables - 1>, 1> result;
    p.for_each([&](scalar const& coefficent, monomial_indices<variables> const& indices) { result[{ indices[index] }][split(indices, index)] += coefficent; });
    return result;
}

template <typename scalar, int variables>
polynomial<scalar, variables + 1> unhide_in(polynomial<polynomial<scalar, 1>, variables> const& p, int index)
{
    polynomial<scalar, variables + 1> result;
    p.for_each([&](polynomial<scalar, 1> const& coefficent_a, monomial_indices<variables> const& indices_a) { coefficent_a.for_each([&](scalar const& coefficient_b, monomial_indices<1> const& indices_b) { result[merge(indices_a, index, indices_b)] += coefficient_b; }); });
    return result;
}

template <typename scalar, int variables>
polynomial<scalar, variables + 1> unhide_out(polynomial<polynomial<scalar, variables>, 1> const& p, int index)
{
    polynomial<scalar, variables + 1> result;
    p.for_each([&](polynomial<scalar, variables> const& coefficent_a, monomial_indices<1> const& indices_a) { coefficent_a.for_each([&](scalar const& coefficient_b, monomial_indices<variables> const& indices_b) { result[merge(indices_b, index, indices_a)] += coefficient_b; }); });
    return result;
}

template<typename scalar, int variables>
polynomial<scalar, variables> substitute(polynomial<scalar, variables> const& p, monomial_mask<variables> const& mask, std::array<scalar, variables> const& values)
{
    polynomial<scalar, variables> result;
    monomial_powers<scalar, variables> powers(values);
    p.for_each([&](scalar const& coefficent, monomial_indices<variables> const& indices) { result[select(indices, mask, true)] += coefficent * powers[select(indices, mask, false)]; });
    return result;
}

//=============================================================================
// remove_polynomial
//=============================================================================

template <typename scalar, typename... Args>
struct remove_polynomial
{
    typedef scalar type;
    typedef std::tuple<Args...> indices;
};

template <typename scalar, int variables, typename... Args>
struct remove_polynomial<polynomial<scalar, variables>, Args...>
{
    typedef typename remove_polynomial<scalar, Args..., monomial_indices<variables>>::type type;
    typedef typename remove_polynomial<scalar, Args..., monomial_indices<variables>>::indices indices;
};

template <typename T>
using remove_polynomial_type = typename remove_polynomial<T>::type;

template <typename T>
using remove_polynomial_indices = typename remove_polynomial<T>::indices;

//=============================================================================
// grevlex_generator
//=============================================================================

/*
* grevlex example for 3 variables (x, y, z) up to total degree 3
*
*index indices monomial
*    0 [0,0,0] 1
*
*    1 [1,0,0] x
*    2 [0,1,0] y
*    3 [0,0,1] z
*
*    4 [2,0,0] x^2
*    5 [1,1,0] x*y
*    6 [1,0,1] x*z
*    7 [0,2,0] y^2
*    8 [0,1,1] y*z
*    9 [0,0,2] z^2
*
*   10 [3,0,0] x^3
*   11 [2,1,0] x^2*y
*   12 [2,0,1] x^2*z
*   13 [1,2,0] x*y^2
*   14 [1,1,1] x*y*z
*   15 [1,0,2] x*z^2
*   16 [0,3,0] y^3
*   17 [0,2,1] y^2*z
*   18 [0,1,2] y*z^2
*   19 [0,0,3] z^3
*/

template <int variables>
class grevlex_generator
{
public:
    //-------------------------------------------------------------------------
    // type
    //-------------------------------------------------------------------------

    enum { variables_length = variables};
    using monomial_indices_type = monomial_indices<variables>;

private:
    //-------------------------------------------------------------------------
    // data
    //-------------------------------------------------------------------------

    monomial_indices_type indices;
    int power;
    int sum;
    int index;

    //-------------------------------------------------------------------------
    // constructors
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // sequence
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // status
    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------
    // compare
    //-------------------------------------------------------------------------

    static bool compare(monomial_indices_type const& a, monomial_indices_type const& b)
    {
        int ap = total_degree(a);
        int bp = total_degree(b);

        if (ap != bp) { return ap < bp; }
        for (int i = 0; i < variables; ++i) { if (a[i] > b[i]) { return true; } else if (a[i] < b[i]) { return false; } }
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
// division
//=============================================================================

template <typename generator, typename scalar, int variables>
monomial<scalar, variables> leading_term(polynomial<scalar, variables> const& p)
{
    monomial<scalar, variables> result = p[{}];
    p.for_each([&](scalar const& coefficent, monomial_indices<variables> const& indices) { if (generator::compare(result.indices, indices)) { result = { coefficent, indices }; } });
    return result;
}

template <typename generator, typename scalar, int variables>
void sort(monomial_vector<scalar, variables>& monomials, bool descending)
{
    std::sort(monomials.begin(), monomials.end(), [&](monomial<scalar, variables> const& a, monomial<scalar, variables> const& b) { return generator::compare(a.indices, b.indices) - descending; });
}

template <typename scalar, int variables>
polynomial<scalar, variables> simple_reduce(polynomial<scalar, variables> const& f, polynomial<scalar, variables> const& g, monomial<scalar, variables> const& mt_f, monomial<scalar, variables> const& mt_g)
{
    return (mt_g * f) - (mt_f * g);
}

template <typename scalar, int variables>
polynomial<scalar, variables> s_polynomial(polynomial<scalar, variables> const& f, polynomial<scalar, variables> const& g, monomial<scalar, variables> const& lt_f, monomial<scalar, variables> const& lt_g)
{
    monomial_indices<variables> gcd_fg = gcd(lt_f.indices, lt_g.indices);
    return simple_reduce(f, g, { lt_f.coefficient, lt_f.indices - gcd_fg }, { lt_g.coefficient, lt_g.indices - gcd_fg });
}

template <typename scalar, int variables>
struct result_reduce
{
    polynomial<scalar, variables> quotient;
    polynomial<scalar, variables> remainder;
    bool irreducible;
};

template <typename scalar, int variables>
result_reduce<scalar, variables> reduce(polynomial<scalar, variables> const& dividend, polynomial<scalar, variables> const& divisor, monomial<scalar, variables> const& mt_dividend, monomial<scalar, variables> const& lt_divisor)
{
    monomial<scalar, variables> q = monomial(mt_dividend.coefficient, mt_dividend.indices - lt_divisor.indices);
    if (is_integral(q.indices)) { return { q, simple_reduce(dividend, divisor, q, { lt_divisor.coefficient, {} }), false }; } else { return { 0, dividend, true }; }
}

//=============================================================================
// matrix of polynomials
//=============================================================================

// input: Matrix MxN, interpreted as vector of MxN monomials [A(0,0),A(1,0)...A(M-2,N-1),A(M-1,N-1)] in grevlex order
// output: polynomial
template <typename scalar, int variables, typename A>
polynomial<scalar, variables> matrix_to_polynomial_grevlex(Eigen::DenseBase<A> const& src)
{
    polynomial<scalar, variables> dst;
    grevlex_generator<variables> gg;
    for (int i = 0; i < src.size(); ++i) { dst[gg.next().current_indices()] = src(i); }
    return dst;
}

// input: Matrix MxN for M equations with N monomials in grevlex order
// output: Matrix of polynomial PxQ with PxQ=M filled in order A(0,0),A(1,0),A(2,0)...A(P-2,Q-1),A(P-1,Q-1)
template <typename scalar, int variables, int _rows, int _cols, typename A>
Eigen::Matrix<polynomial<scalar, variables>, _rows, _cols> matrix_to_polynomial_grevlex(Eigen::MatrixBase<A> const& M, int rows = _rows, int cols = _cols)
{
    Eigen::Matrix<polynomial<scalar, variables>, _rows, _cols> dst(rows, cols);
    for (int i = 0; i < cols; ++i) { for (int j = 0; j < rows; ++j) { dst(j, i) = matrix_to_polynomial_grevlex<scalar, variables>(M.row((i * rows) + j)); } }
    return dst;
}

// input: polynomial
// output: Matrix MxN filled as vector in grevlex order A(0,0),A(1,0)...A(M-2,N-1),A(M-1,N-1), intended for building 1xN or Mx1 monomial vectors
template <typename _matrix_scalar, int _rows, int _cols, typename scalar, int variables>
Eigen::Matrix<_matrix_scalar, _rows, _cols> matrix_from_polynomial_grevlex(polynomial<scalar, variables> const& src, int rows = _rows, int cols = _cols)
{
    Eigen::Matrix<_matrix_scalar, _rows, _cols> dst(rows, cols);// = Eigen::Matrix<_matrix_scalar, _rows, _cols>::Zero(rows, cols);
    grevlex_generator<variables> gg;
    for (int i = 0; i < dst.size(); ++i) { dst(i) = src[gg.next().current_indices()]; }
    //for (int i = 0; i < cols; ++i) {
    //    for (int j = 0; j < rows; ++j) {
    //        dst(j, i) = src[gg.next().current_indices()];
    //    }
    //}
    /*
    src.for_each
    (
        [&](scalar const& element, monomial_indices<variables> const& indices)
        {
            int i = grevlex_generator<variables>::ravel(indices);
            if (i < dst.size()) { dst(i) = element; }
        }
    );
    */
    return dst;
}

// input: Matrix MxN of polynomials
// output: Matrix QxP of monomials with Q=MxN and P maximum monomials to extract from polynomials in grevlex order
template <typename _matrix_scalar, int _rows, int _cols, typename A>
Eigen::Matrix<_matrix_scalar, _rows, _cols> matrix_from_polynomial_grevlex(Eigen::MatrixBase<A> const& src, int rows = _rows, int cols = _cols)
{
    Eigen::Matrix<_matrix_scalar, _rows, _cols> dst(rows, cols);
    Eigen::Index input_rows = src.rows();
    Eigen::Index input_cols = src.cols();
    for (int i = 0; i < input_cols; ++i) { for (int j = 0; j < input_rows; ++j) { dst.row((i * input_rows) + j) = matrix_from_polynomial_grevlex<_matrix_scalar, 1, _cols>(src(j, i), 1, cols); } }
    return dst;
}

//=============================================================================
// matrix operations
//=============================================================================

template <typename scalar, int variables, int _rows, int _cols>
Eigen::Matrix<polynomial<polynomial<scalar, 1>, variables - 1>, _rows, _cols> hide_in(Eigen::Matrix<polynomial<scalar, variables>, _rows, _cols> const& src, int index)
{
    Eigen::Matrix<polynomial<polynomial<scalar, 1>, variables - 1>, _rows, _cols> dst(src.rows(), src.cols());
    for (int i = 0; i < src.cols(); ++i) { for (int j = 0; j < src.rows(); ++j) { dst(j, i) = hide_in(src(j, i), index); } }
    return dst;
}

template <typename scalar, int variables, int _rows, int _cols>
Eigen::Matrix<polynomial<polynomial<scalar, variables - 1>, 1>, _rows, _cols> hide_out(Eigen::Matrix<polynomial<scalar, variables>, _rows, _cols> const& src, int index)
{
    Eigen::Matrix<polynomial<polynomial<scalar, variables - 1>, 1>, _rows, _cols> dst(src.rows(), src.cols());
    for (int i = 0; i < src.cols(); ++i) { for (int j = 0; j < src.rows(); ++j) { dst(j, i) = hide_out(src(j, i), index); } }
    return dst;
}

template <typename scalar, int variables, int _rows, int _cols>
Eigen::Matrix<polynomial<scalar, variables + 1>, _rows, _cols> unhide_in(Eigen::Matrix<polynomial<polynomial<scalar, 1>, variables>, _rows, _cols> const& src, int index)
{
    Eigen::Matrix<polynomial<scalar, variables + 1>, _rows, _cols> dst(src.rows(), src.cols());
    for (int i = 0; i < src.cols(); ++i) { for (int j = 0; j < src.rows(); ++j) { dst(j, i) = unhide_in(src(j, i), index); } }
    return dst;
}

template <typename scalar, int variables, int _rows, int _cols>
Eigen::Matrix<polynomial<scalar, variables + 1>, _rows, _cols> unhide_out(Eigen::Matrix<polynomial<polynomial<scalar, variables>, 1>, _rows, _cols> const& src, int index)
{
    Eigen::Matrix<polynomial<scalar, variables + 1>, _rows, _cols> dst(src.rows(), src.cols());
    for (int i = 0; i < src.cols(); ++i) { for (int j = 0; j < src.rows(); ++j) { dst(j, i) = unhide_out(src(j, i), index); } }
    return dst;
}

template <typename scalar, int _rows, int _cols, int variables>
Eigen::Matrix<polynomial<scalar, variables>, _rows, _cols> substitute(Eigen::Matrix<polynomial<scalar, variables>, _rows, _cols> const& src, monomial_mask<variables> const& mask, std::array<scalar, variables> const& values)
{
    Eigen::Matrix<polynomial<scalar, variables>, _rows, _cols> dst(src.rows(), src.cols());
    for (int i = 0; i < src.cols(); ++i) { for (int j = 0; j < src.rows(); ++j) { dst(j, i) = substitute(src(j, i), mask, values); } }
    return dst;
}

template <typename scalar, int _rows, int _cols, int variables>
void substitute_inplace(Eigen::Matrix<polynomial<scalar, variables>, _rows, _cols>& src, monomial_mask<variables> const& mask, std::array<scalar, variables> const& values)
{
    for (int i = 0; i < src.cols(); ++i) { for (int j = 0; j < src.rows(); ++j) { src(j, i) = substitute(src(j, i), mask, values); } }
}


// TODO: functions for polynomial -> matrix of polynomials

template <typename scalar, int _rows, int _cols, int variables>
Eigen::Matrix<scalar, _rows, _cols> split(Eigen::Matrix<polynomial<scalar, variables>, _rows, _cols> const& src, monomial_indices<variables> const& indices)
{
    Eigen::Matrix<scalar, _rows, _cols> dst(src.rows(), src.cols());
    for (int i = 0; i < src.cols(); ++i) { for (int j = 0; j < src.rows(); ++j) { dst(j, i) = src(j, i)[indices]; } }
    return dst;
}
