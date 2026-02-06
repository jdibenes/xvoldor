
#include <Eigen/Eigen>
#include <type_traits>
#include <iostream>
#include "helpers_eigen.h"
#include "helpers_geometry.h"









std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> vector_matrix_multiply(std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> const& a, int rows_a, int cols_a, std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> const& b, int rows_b, int cols_b, bool transpose_a, bool transpose_b)
{
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> c;

    int ar;
    int ac;
    if (transpose_a) {
        ar = cols_a;
        ac = rows_a;
    }
    else {
        ar = rows_a;
        ac = cols_a;
    }

    int br;
    int bc;
    if (transpose_b) {
        br = cols_b;
        bc = rows_b;
    }
    else {
        br = rows_b;
        bc = cols_b;
    }

    // ac == br

    for (int row_a = 0; row_a < ar; ++row_a) {
        for (int col_b = 0; col_b < bc; ++col_b) {
            int idx_c = 0;
            for (int e = 0; e < ac; ++e) {
                int idx_a = 0;
                int idx_b = 0;

                c[idx_c] = vector_add(c[idx_c], vector_multiply(a[idx_a], b[idx_b]));
            }
        }
    }










}







//1           2           3
//std::vector<std::vector<std::vector<float>>>
//std::vector<T>

//using StorageType




template <typename T>
class multivariate_polynomial
{
private:
    T data;
    std::vector<int> indices;

    template <typename T_intermediate>
    int set_variables(T_intermediate& object, int level)
    {
        if constexpr (std::is_arithmetic_v<T_intermediate>)
        {
            return level;
        }
        else
        {
            object.resize(1);
            return set_variables(object[0], level + 1);
        }
    }

    template <typename T_intermediate>
    auto& bias(T_intermediate& object)
    {
        if constexpr (std::is_arithmetic_v<T_intermediate>)
        {
            return object;
        }
        else
        {
            return bias(object[0]);
        }
    }

    template <typename T_intermediate>
    auto const& bias(T_intermediate const& object) const
    {
        if constexpr (std::is_arithmetic_v<T_intermediate>)
        {
            return object;
        }
        else
        {
            return bias(object[0]);
        }
    }

    template <typename T_intermediate, typename T_final>
    void for_each(T_intermediate& object, int level, std::vector<int>& indices, std::function<void(T_final&, std::vector<int> const&)> callback)
    {
        if constexpr (std::is_arithmetic_v<T_intermediate>)
        {
            callback(object, indices);
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

    template <typename T_intermediate, typename T_final>
    void for_each(T_intermediate const& object, int level, std::vector<int>& indices, std::function<void(T_final const&, std::vector<int> const&)> callback) const
    {
        if constexpr (std::is_arithmetic_v<T_intermediate>)
        {
            callback(object, indices);
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

    template <typename T_intermediate>
    auto& at(T_intermediate& object, int level, std::vector<int> const& indices)
    {
        if constexpr (std::is_arithmetic_v<T_intermediate>)
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

    template <typename T_intermediate>
    auto const& at(T_intermediate const& object, int level, std::vector<int> const& indices) const
    {
        if constexpr (std::is_arithmetic_v<T_intermediate>)
        {
            return object;
        }
        else
        {
            int index = indices[level];
            return at(object.at(index), level + 1, indices);
        }
    }

public:
    multivariate_polynomial()
    {
        indices.resize(set_variables(data, 0));
    }

    template <typename T_final>
    void for_each(std::function<void(T_final&, std::vector<int> const&)> callback)
    {
        for_each(data, 0, indices, callback);
    }

    template <typename T_final>
    void for_each(std::function<void(T_final const&, std::vector<int> const&)> callback) const
    {
        std::vector<int> _indices{ indices };
        for_each(data, 0, _indices, callback);
    }

    auto& bias() 
    {
        return bias(data);
    }

    auto const& bias() const
    {
        return bias(data);
    }

    auto& at(std::vector<int> const& indices)
    {
        return at(data, 0, indices);
    }

    auto const& at(std::vector<int> const& indices) const
    {
        return at(data, 0, indices);
    }

    auto& operator[](std::vector<int> const& indices)
    {
        return at(indices);
    }

    auto const& operator[](std::vector<int> const& indices) const
    {
        return at(indices);
    }

    multivariate_polynomial<T> operator+() const
    {
        return *this;
    }

    multivariate_polynomial<T> operator+(multivariate_polynomial<T> const& other) const
    {
        multivariate_polynomial<T> result = *this;
        std::function<void(float const&, std::vector<int> const&)> f([&](float const& element, std::vector<int> const& indices) { result.at(indices) += element; });
        other.for_each(f);
        return result;
    }

    template<typename A>
    multivariate_polynomial<T> operator+(A const& other)
    {
        multivariate_polynomial<T> result = *this;
        result.bias() += other;
        return result;
    }

    multivariate_polynomial<T> operator-() const
    {
        multivariate_polynomial<T> result = *this;
        std::function<void(float&, std::vector<int> const&)> f([&](float& element, std::vector<int> const& indices) { element = -element; });
        result.for_each(f);
        return result;
    }

    multivariate_polynomial<T> operator-(multivariate_polynomial<T> const& other) const
    {
        multivariate_polynomial<T> result = *this;
        std::function<void(float const&, std::vector<int> const&)> f([&](float const& element, std::vector<int> const& indices) { result.at(indices) -= element; });
        other.for_each(f);
        return result;
    }

    template<typename A>
    multivariate_polynomial<T> operator-(A const& other)
    {
        multivariate_polynomial<T> result = *this;
        result.bias() -= other;
        return result;
    }

    multivariate_polynomial<T> operator*(multivariate_polynomial<T> const& other) const
    {
        multivariate_polynomial<T> result;
        std::vector<int> indices_c{ indices };

        std::function<void(float const&, std::vector<int> const&)> f
        (
            [&](float const& element_a, std::vector<int> const& indices_a)
            {
                std::function<void(float const&, std::vector<int> const&)> g
                (
                    [&](float const& element_b, std::vector<int> const indices_b)
                    {
                        for (int i = 0; i < indices_b.size(); ++i) { indices_c[i] = indices_a[i] + indices_b[i]; }
                        result.at(indices_c) = element_a * element_b;
                    }
                );
                other.for_each(g);
            }
        );
        for_each(f);

        return result;
    }

    template<typename A>
    multivariate_polynomial<T> operator*(A const& other) const
    {
        multivariate_polynomial<T> result = *this;
        std::function<void(float&, std::vector<int> const&)> f([&](float& element, std::vector<int> const& indices) { element *= other; });
        result.for_each(f);
        return result;
    }









    //+, -, *, /






};




void test_poly()
{
    multivariate_polynomial<std::vector<std::vector<float>>> p;
    std::function<void(float const&, std::vector<int> const&)> f([](float const& element, std::vector<int> const& indices) { std::cout << indices[0] << ":" << indices[1] << " -> " << element << std::endl; /*element += 14.0f*/; });
    p.for_each(f);
    p.for_each(f);
    p.for_each(f);
    auto const& x = p[{0, 0}];


    multivariate_polynomial<std::vector<std::vector<float>>> const& p_const = p;

    float y = p_const.at({ 0,0 });


    p.at({ 1, 2 }) = 12;
    float z = p.at({ 1, 2 });
    std::cout << "Z: " << z << std::endl;

    std::vector<float> t;

    multivariate_polynomial<std::vector<std::vector<float>>> A;
    multivariate_polynomial<std::vector<std::vector<float>>> B;

    A.at({ 1, 1 }) = 5; // 5xy
    B.at({ 1, 1 }) = 2; // 2xy
    B.at({ 1, 0 }) = 3; // 3x

    A.for_each(f);
    B.for_each(f);
    auto C = A * B; // 5xy * (2xy + 3x) = 10x^2y^2 + 15x^2y -> {2, 2} -> 10, {2, 1} -> 15
    C.for_each(f);

    auto D = A + 3;
    D.for_each(f);

    auto E = B * 11;
    E.for_each(f);


    //t.resize();
    //p.data.resize(5);
    //p.data[2].resize(7);
    //p.data[4].resize(3);


    
    //std::vector<int> indices{ 0,0 };

    //p.for_each(p.data, 0, indices, f);





}







/*
float& at(std::vector<int> indices)
{
}

float& at(std::vector<int> degrees)
{
    void*& base = root;
    int last = variables - 1;
    for (int i = 0; i < last; ++i)
    {
        int n = degrees[i];
        if (base == nullptr) { base = new std::vector<void*>(); }
        std::vector<void*>* current = static_cast<std::vector<void*>*>(base);
        if (n >= current->size()) { current->resize(n + 1, nullptr); }
        base = (*current)[n];
    }
    int n = degrees[last];
    if (base == nullptr) { base = new std::vector<float>(); }
    std::vector<float>* current = static_cast<std::vector<float>*>(base);
    if (n >= current->size()) { current->resize(n + 1, 0); }
    return (*current)[n];
}
*/

class poly
{
private:
    void* root;
    int variables;


    void delete_node(void*& root, int level)
    {
        if (root == nullptr) { return; }
        if (level >= (this->variables - 1)) {
            delete static_cast<std::vector<float>*>(root);
            root = nullptr;
            return;
        }
        std::vector<void*>* current = static_cast<std::vector<void*>*>(root);
        for (auto& next : *current)
        {
            delete_node(next, level + 1);
        }
        delete current;
        root = nullptr;
    }

public:
    poly(int variables)
    {
        this->variables = variables;
        this->root = nullptr;
    }

    virtual ~poly()
    {
        delete_node(this->root, 0);
    }


    float peek(std::vector<int> degrees) const
    {
        void* base = root;
        int last = variables - 1;
        for (int i = 0; i < last; ++i)
        {
            int n = degrees[i];
            if (base == nullptr) { return 0; }
            std::vector<void*>* current = static_cast<std::vector<void*>*>(base);
            if (n >= current->size()) { return 0; }
            base = (*current)[n];
        }
        int n = degrees[last];
        if (base == nullptr) { return 0; }
        std::vector<float>* current = static_cast<std::vector<float>*>(base);
        if (n >= current->size()) { return 0; }
        return (*current)[n];
    }

    float& at(std::vector<int> degrees)
    {
        void*& base = root;
        int last = variables - 1;
        for (int i = 0; i < last; ++i)
        {
            int n = degrees[i];
            if (base == nullptr) { base = new std::vector<void*>(); }
            std::vector<void*>* current = static_cast<std::vector<void*>*>(base);
            if (n >= current->size()) { current->resize(n + 1, nullptr); }
            base = (*current)[n];
        }
        int n = degrees[last];
        if (base == nullptr) { base = new std::vector<float>(); }
        std::vector<float>* current = static_cast<std::vector<float>*>(base);
        if (n >= current->size()) { current->resize(n + 1, 0); }
        return (*current)[n];
    }

    poly operator+(poly const& other)
    {
        poly result(this->variables);




        //for (int i = 0; )


    }

    poly operator-(poly const& other)
    {

    }

    poly operator*(poly const& other)
    {

    }
};
/*
*/





bool solver_gpm_nm6(float const* p1, float const* p2, float* r01, float* t01)
{
    /*
    Eigen::Matrix<float, 3, 6> P1 = matrix_from_buffer<float, 3, 7>(p1);
    Eigen::Matrix<float, 3, 6> P2 = matrix_from_buffer<float, 3, 7>(p2);

    Eigen::Matrix<float, 2, 6> q1 = P1.colwise().hnormalized();
    Eigen::Matrix<float, 2, 6> q2 = P2.colwise().hnormalized();

    Eigen::Matrix<float, 3, 6> Q1 = q1.colwise().homogeneous();
    Eigen::Matrix<float, 3, 6> Q2 = q2.colwise().homogeneous();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q = matrix_E_constraints(Q1, Q2);

    Q.col(4) = Q.col(4) - Q.col(0);
    Q.col(8) = Q.col(8) - Q.col(0);

    Eigen::Matrix<float, 8, 2> k = Q(Eigen::all, Eigen::seqN(1, 8)).fullPivLu().kernel();
    Eigen::Matrix<float, 9, 2> e;

    e << (-(k(3, Eigen::all) + k(7, Eigen::all))), k;

    std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> v;

    for (int i = 0; i < 9; ++i) { v.push_back(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>{ e(0, Eigen::all) }); }






    





    






    




    // det(E) == 0
    // 2*E*E^T*E - tr(E*E^T)E











    
    

    Eigen::Matrix<float, 3, 3> fake_E = e.reshaped(3, 3);

    result_R_t_from_E result = R_t_from_E(fake_E, q1, q2);

    Eigen::Matrix<float, 3, 3> R = result.P(Eigen::all, Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 1> v = result.P.col(3);

    Eigen::AngleAxis<float> aa(R);

    Eigen::Matrix<float, 3, 1> r = aa.axis() * aa.angle();
    Eigen::Matrix<float, 3, 1> t = (P2.col(0) - R * P1.col(0)).norm() * v;

    matrix_to_buffer(r, r01);
    matrix_to_buffer(t, t01);

    float r_sum = r01[0] + r01[1] + r01[2];
    float t_sum = t01[0] + t01[1] + t01[2];

    float x_sum = r_sum + t_sum;

    return std::isfinite(x_sum);
    */
        return true;
}
