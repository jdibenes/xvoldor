
#include <Eigen/Eigen>
#include <iostream>
#include "polynomial.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"

template <typename _scalar, int _rows, int _cols, int _n>
void polynomial_row_echelon_sort(Eigen::Matrix<polynomial<_scalar, _n>, _rows, _cols>& M, int row, int col, monomial_indices_t const& monomial)
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
bool polynomial_row_echelon_eliminate(Eigen::Matrix<polynomial<_scalar, _n>, _rows, _cols>& M, int row, int col, monomial_indices_t const& monomial, bool all)
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
bool polynomial_row_echelon_step(Eigen::Matrix<polynomial<_scalar, _n>, _rows, _cols>& M, int row, int col, monomial_indices_t const& monomial, bool all)
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

    for (int i = 0; i < 4; ++i) { polynomial_row_echelon_step(H, i, i, std::vector{ 0 }, false); }

    Eigen::Matrix<polynomial<float, 1>, 6, 6> D = H(Eigen::seqN(4, 6), Eigen::seqN(4, 6));

    polynomial_row_echelon_step(D, 0, 0, std::vector{ 1 }, true);
    polynomial_row_echelon_step(D, 1, 0, std::vector{ 0 }, true);

    polynomial_row_echelon_step(D, 2, 1, std::vector{ 1 }, true);
    polynomial_row_echelon_step(D, 3, 1, std::vector{ 0 }, true);

    polynomial_row_echelon_step(D, 4, 2, std::vector{ 1 }, true);
    polynomial_row_echelon_step(D, 5, 2, std::vector{ 0 }, true);

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

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            D(i, j).for_each
            (
                [](float const& element, monomial_indices_t const& indices)
                {
                    if (element != 0)
                    {
                        std::cout << element << "*z^" << indices[0] << " + ";
                    }
                }
            );
            std::cout << " || ";
        }
        std::cout << std::endl;
    }







    















    //S.rowwise().reverseInPlace();



    





    

    

    




    



    /*
    polynomial<float, 1> H_determinant = H(4,4);
    for (int i = 5; i < 10; ++i)
    {
        H_determinant *= H(i, i);
    }

    H_determinant.for_each([](float const& element, monomial_indices_t const& indices) { std::cout << element << "*z^" << indices[0] << ","; });
    std::cout << std::endl;
    */
         
    return false;
         
         
         

    //Eigen::Matrix<float, 10, 1> p_z = matrix_from_polynomial_grevlex<float, 10, 1>(H.determinant());


    //H(Eigen::seqN(0, 3), Eigen::seqN(0, 3));










    //Eigen::Matrix<float, 9, 3> yx1 = e(Eigen::all, Eigen::seqN(0, 3));
    //Eigen::Matrix<float, 9, 2> z0;






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