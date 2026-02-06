
#include <Eigen/Eigen>
#include <iostream>
#include "polynomial.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"


template <typename _scalar, int _n>
class polynomial_matrix_3x3
{
private:
    polynomial<_scalar, _n> data[3][3];


public:
    polynomial()
    {
    }

    polynomial<_scalar, _n> determinant()
    {
        return data[0][0] * (data[1][1] * data[2][2] - data[1][2] * data[2][1])
             - data[0][1] * (data[1][0] * data[2][2] - data[1][2] * data[2][0])
             + data[0][2] * (data[1][0] * data[2][1] - data[1][1] * data[2][0]);
    }
};





bool solver_gpm_nm6(float const* p1, float const* p2, float* r01, float* t01)
{
    Eigen::Matrix<float, 3, 6> P1 = matrix_from_buffer<float, 3, 6>(p1);
    Eigen::Matrix<float, 3, 6> P2 = matrix_from_buffer<float, 3, 6>(p2);

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

    // e = [e11 e21 e31 e12 e22 e32 e13 e23 e33]

    // det(E) == 0
    // 2*E*E^T*E - tr(E*E^T)E

    polynomial<float, 1> e_poly;
    e0[{0}] = e(0, 0);
    e0[{1}] = e(0, 1);






    //std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> v;

    //for (int i = 0; i < 9; ++i) { v.push_back(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>{ e(0, Eigen::all) }); }
























    














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




void test_poly()
{
    polynomial<float, 3> A;
    polynomial<float, 3> B;

    A[{ 0, 0, 0 }] = 2;
    A[{ 1, 2, 0 }] = 4;
    A[{ 3, 1, 2 }] = 3;

    B[{ 1, 0, 0 }] = 5;
    B[{ 2, 2, 2 }] = 6;

    polynomial<float, 3> const& Ar = A;

    float a = A[{ 0, 0, 0 }];
    float ar = Ar[{ 0, 0, 0 }];
    //Ar.at({ 0,0,0 }) = 11;
    

    std::function<void(float const&, std::vector<int> const&)> f([](float const& element, std::vector<int> const& indices) { std::cout << indices[0] << ":" << indices[1] << ":" << indices[2] << " -> " << element << std::endl; /*element += 14.0f*/; });
    std::function<void(float&, std::vector<int> const&)> g([](float const& element, std::vector<int> const& indices) { std::cout << indices[0] << ":" << indices[1] << ":" << indices[2] << " -> " << element << std::endl; /*element += 14.0f*/; });

    //f = g;
    //g = f;






    polynomial<float, 3> C = A + B;
    C = A - B;
    C = +A;
    C = -A;
    C = A * B;
    C = A * 2;
    C = 2 * A;
    C += A;
    C -= B;
    C *= A;
    C *= -2;

    polynomial<double, 3> D = C.cast<double>();






    A.for_each(f);
    Ar.for_each(f);
    C.for_each(f);

    A.for_each(g);
    //Ar.for_each(g);
    C.for_each(g);
    

    
    //multivariate_polynomial<std::vector<std::vector<float>>> p;
    //std::function<void(float const&, std::vector<int> const&)> f([](float const& element, std::vector<int> const& indices) { std::cout << indices[0] << ":" << indices[1] << " -> " << element << std::endl; /*element += 14.0f*/; });
    /*
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
    */

    //t.resize();
    //p.data.resize(5);
    //p.data[2].resize(7);
    //p.data[4].resize(3);


    
    //std::vector<int> indices{ 0,0 };

    //p.for_each(p.data, 0, indices, f);





}

