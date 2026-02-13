
#include <Eigen/Eigen>
#include <iostream>
#include "polynomial.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"
#include "../thirdparty/rnp/sturm.h"
/*
//NO
template <typename kind, int count>
struct objkindn
{
    using kind = kind;
    enum { count = count };
};
*/



template <typename _scalar, int _rows, int _cols, int _n>
void polynomial_row_echelon_sort(Eigen::Matrix<polynomial<_scalar, _n>, _rows, _cols>& M, int row, int col, monomial_indices<_n> const& monomial)
{
    _scalar max_s = _scalar(0);
    int max_r = row;

    for (int i = row; i < M.rows(); ++i)
    {
        _scalar v = abs(M(i, col)[monomial]);
        if (v <= max_s) { continue; }
        max_s = v;
        max_r = i;
    }

    if (max_r != row) { M.row(row).swap(M.row(max_r)); }
}

template <typename _scalar, int _rows, int _cols, int _n>
bool polynomial_row_echelon_eliminate(Eigen::Matrix<polynomial<_scalar, _n>, _rows, _cols>& M, int row, int col, monomial_indices<_n> const& monomial, bool all)
{
    _scalar a = M(row, col)[monomial];
    if (abs(a) <= 0) { return false; }
    M.row(row) /= a;

    for (int i = all ? 0 : (row + 1); i < M.rows(); ++i)
    {
        if (i == row) { continue; }
        _scalar b = M(i, col)[monomial];
        if (abs(b) <= 0) { continue; }
        M.row(i) -= b * M.row(row);
    }

    return true;
}

template <typename _scalar, int _rows, int _cols, int _n>
bool polynomial_row_echelon_step(Eigen::Matrix<polynomial<_scalar, _n>, _rows, _cols>& M, int row, int col, monomial_indices<_n> const& monomial, bool all)
{
    polynomial_row_echelon_sort(M, row, col, monomial);
    return polynomial_row_echelon_eliminate(M, row, col, monomial, all);
}

bool solver_rpe_easy(float const* p1, float const* p2, float* r01, float* t01)
{
    Eigen::Matrix<float, 3, 5> P1 = matrix_from_buffer<float, 3, 5>(p1);
    Eigen::Matrix<float, 3, 5> P2 = matrix_from_buffer<float, 3, 5>(p2);

    Eigen::Matrix<float, 2, 5> q1 = P1.colwise().hnormalized();
    Eigen::Matrix<float, 2, 5> q2 = P2.colwise().hnormalized();

    Eigen::Matrix<float, 3, 5> Q1 = q1.colwise().homogeneous();
    Eigen::Matrix<float, 3, 5> Q2 = q2.colwise().homogeneous();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q = matrix_E_constraints(Q1, Q2);

    Eigen::Matrix<float, 9, 4> e = Q.fullPivLu().kernel();

    Eigen::Matrix<polynomial<float, 3>, 3, 3> E = matrix_to_polynomial_grevlex<float, 3, 3, 3>(e); // OK

    polynomial<float, 3> E_determinant = E.determinant();

    Eigen::Matrix<polynomial<float, 3>, 3, 3> EEt = E * E.transpose();
    Eigen::Matrix<polynomial<float, 3>, 3, 3> E_singular_values = (EEt * E) - ((0.5 * EEt.trace()) * E);

    Eigen::Matrix<float, 10, 20> S;

    S << matrix_from_polynomial_grevlex<float, 9, 20>(E_singular_values),
         matrix_from_polynomial_grevlex<float, 1, 20>(E_determinant);

    std::cout << "S" << std::endl;
    std::cout << S << std::endl;

    /*
    * hide z
    *
    *  0 [0,0,0] 1      3 [0,0,1] z      9 [0,0,2] z^2   19 [0,0,3] z^3
    *  1 [1,0,0] x      6 [1,0,1] x*z   15 [1,0,2] x*z^2
    *  2 [0,1,0] y      8 [0,1,1] y*z   18 [0,1,2] y*z^2
    *  4 [2,0,0] x^2   12 [2,0,1] x^2*z
    *  5 [1,1,0] x*y   14 [1,1,1] x*y*z
    *  7 [0,2,0] y^2   17 [0,2,1] y^2*z
    * 10 [3,0,0] x^3
    * 11 [2,1,0] x^2*y
    * 13 [1,2,0] x*y^2
    * 16 [0,3,0] y^3
    */

    Eigen::Matrix<polynomial<float, 1>, 10, 10> H;

    H << matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all,   16)),
         matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all,   13)),
         matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all,   11)),
         matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all,   10)),
         matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, {  7, 17 })),
         matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, {  5, 14 })),
         matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, {  4, 12 })),
         matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, {  2,  8, 18 })),
         matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, {  1,  6, 15 })),
         matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, {  0,  3,  9, 19 }));

    for (int i = 0; i < 4; ++i) { polynomial_row_echelon_step(H, i, i, { 0 }, false); }

    Eigen::Matrix<polynomial<float, 1>, 6, 6> D = H(Eigen::seqN(4, 6), Eigen::seqN(4, 6));

    polynomial_row_echelon_step(D, 0, 0, { 1 }, true);
    polynomial_row_echelon_step(D, 1, 0, { 0 }, true);

    polynomial_row_echelon_step(D, 2, 1, { 1 }, true);
    polynomial_row_echelon_step(D, 3, 1, { 0 }, true);

    polynomial_row_echelon_step(D, 4, 2, { 1 }, true);
    polynomial_row_echelon_step(D, 5, 2, { 0 }, true);

    polynomial<float, 1> z = create_polynomial_grevlex<float, 1>({0, 1});
    
    D.row(1) = (D.row(1) * z) - D.row(0);
    D.row(3) = (D.row(3) * z) - D.row(2);
    D.row(5) = (D.row(5) * z) - D.row(4);

    D.row(1).swap(D.row(2));
    D.row(2).swap(D.row(4));

    Eigen::Matrix<polynomial<float, 1>, 3, 3> Z = D(Eigen::seqN(3, 3), Eigen::seqN(3, 3));

    polynomial<float, 1> z_poly = Z.determinant();

    Eigen::Matrix<double, 1, 11> z_poly_coef = matrix_from_polynomial_grevlex<float, 1, 11>(z_poly).cast<double>().eval();

    std::cout << "z_poly_coef" << std::endl;
    std::cout << z_poly_coef << std::endl;

    double z_roots[10];
    int n_roots = 0;

    if (!find_real_roots_sturm(z_poly_coef.data(), 10, z_roots, &n_roots, 10, 0))
    {
        return false;
    }

    Eigen::Matrix<float, 10, 10> FINAL;
    Eigen::Matrix<float, 10, 1> solution;
    Eigen::Matrix<float, 3, 3> fake_E;

    std::cout << "roots: " << std::endl;
    for (int i = 0; i < n_roots; ++i)
    {
        

        float z = z_roots[i];
        float z2 = z_roots[i] * z_roots[i];
        float z3 = z_roots[i] * z_roots[i] * z_roots[i];

        FINAL <<  S(Eigen::all, 16),
              S(Eigen::all, 13),
              S(Eigen::all, 11),
              S(Eigen::all, 10),
             (S(Eigen::all,  7) + z * S(Eigen::all, 17)),
             (S(Eigen::all,  5) + z * S(Eigen::all, 14)),
             (S(Eigen::all,  4) + z * S(Eigen::all, 12)),
             (S(Eigen::all,  2) + z * S(Eigen::all,  8) + z2 * S(Eigen::all, 18)),
             (S(Eigen::all,  1) + z * S(Eigen::all,  6) + z2 * S(Eigen::all, 15)),
             (S(Eigen::all,  0) + z * S(Eigen::all,  3) + z2 * S(Eigen::all,  9) + z3 * S(Eigen::all, 19));

        solution = FINAL.bdcSvd(Eigen::ComputeFullV).matrixV().col(9);
        float x = solution(8) / solution(9);
        float y = solution(7) / solution(9);

        std::cout << "(" << x << ", " << y << ", " << z_roots[i] << ")" << std::endl;

        fake_E = (e(Eigen::all, 0) + x * e(Eigen::all, 1) + y * e(Eigen::all, 2) + z * e(Eigen::all, 3)).reshaped(3, 3);
    
        result_R_t_from_E result = R_t_from_E(fake_E, q1, q2);

        Eigen::Matrix<float, 3, 3> R = result.P(Eigen::all, Eigen::seqN(0, 3));
        Eigen::Matrix<float, 3, 1> v = result.P.col(3);

        Eigen::Matrix<float, 3, 1> r = vector_r_rodrigues(R);
        Eigen::Matrix<float, 3, 1> t = (P2.col(0) - R * P1.col(0)).norm() * v;
    
        std::cout << "R" << std::endl;
        std::cout << R << std::endl;
        std::cout << "t" << std::endl;
        std::cout << t << std::endl;
    
    
    }
    std::cout << std::endl;


    

    




    
    

    /*
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            D(i, j).for_each
            (
                [](float const& element, monomial_indices<1> const& indices)
                {
                    if (element != 0)
                    {
                        std::cout << element << "*z^" << indices[0] << " + ";
                    }
                    return true;
                }
            );
            std::cout << " || ";
        }
        std::cout << std::endl;
    }

    z_poly.for_each
    (
        [](float const& element, monomial_indices<1> const& indices)
        {
            if (element != 0)
            {
                std::cout << element << "*z^" << indices[0] << " + ";
            }
            return true;
        }
    );
    std::cout << std::endl;
    */

    return false;

    /*
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

   =>

    0 [0,0,0] 1       3 [0,0,1] z      9 [0,0,2] z^2    19 [0,0,3] z^3

    1 [1,0,0] x       6 [1,0,1] x*z   15 [1,0,2] x*z^2
    2 [0,1,0] y       8 [0,1,1] y*z   18 [0,1,2] y*z^2

    4 [2,0,0] x^2    12 [2,0,1] x^2*z
    5 [1,1,0] x*y    14 [1,1,1] x*y*z
    7 [0,2,0] y^2    17 [0,2,1] y^2*z

   10 [3,0,0] x^3
   11 [2,1,0] x^2*y
   13 [1,2,0] x*y^2
   16 [0,3,0] y^3
    */
}



/*
polynomial<float, 1> H_determinant = H(4,4);
for (int i = 5; i < 10; ++i)
{
    H_determinant *= H(i, i);
}

H_determinant.for_each([](float const& element, monomial_indices_t const& indices) { std::cout << element << "*z^" << indices[0] << ","; });
std::cout << std::endl;
*/


//Eigen::Matrix<float, 10, 1> p_z = matrix_from_polynomial_grevlex<float, 10, 1>(H.determinant());


//H(Eigen::seqN(0, 3), Eigen::seqN(0, 3));










//Eigen::Matrix<float, 9, 3> yx1 = e(Eigen::all, Eigen::seqN(0, 3));
//Eigen::Matrix<float, 9, 2> z0;

/*
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
        {
            H(i, j).for_each([](float const& element, monomial_indices_t const& indices) {std::cout << element << "*z^" << indices[0] << " + "; });
            std::cout << " || ";
        }
        std::cout << std::endl;
    }
    */

    //for (auto vv : D)
       // {
        //    std::cout << vv << ", ";
       // }
        //std::cout << std::endl;
        //polynomial<float, 1> a_1;
        //polynomial<float, 1> b_1;


        //std::array<int, 3> b = std::array<int, 3>{0};