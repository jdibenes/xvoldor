
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





    int hidden_variable_index = 1;

    Eigen::Matrix<polynomial<polynomial<float, 1>, 2>, 3, 3> E_hidden = hide_in(E, hidden_variable_index);

    polynomial<polynomial<float, 1>, 2> E_determinant = E_hidden.determinant();

    Eigen::Matrix<polynomial<polynomial<float, 1>, 2>, 3, 3> EEt = E_hidden * E_hidden.transpose();
    Eigen::Matrix<polynomial<polynomial<float, 1>, 2>, 3, 3> E_singular_values = (EEt * E_hidden) - ((0.5 * EEt.trace()) * E_hidden);

    Eigen::Matrix<polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> S(10, 10);

    S << matrix_from_polynomial_grevlex<polynomial<float, 1>, 9, 10>(E_singular_values).rowwise().reverse(),
         matrix_from_polynomial_grevlex<polynomial<float, 1>, 1, 10>(E_determinant).rowwise().reverse();

    ///////
    //S.row(3).swap(S.row(6));
    //S.row(4).swap(S.row(7));
    //S.row(5).swap(S.row(8));
    ///////
    
    for (int i = 0; i < 4; ++i) { polynomial_row_echelon_step(S, i, i, { 0 }, false); }

    std::cout << "S" << std::endl;
    std::cout << S << std::endl;

    Eigen::Matrix<polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> S6 = S(Eigen::seqN(4, 6), Eigen::seqN(4, 6));

    polynomial_row_echelon_step(S6, 0, 0, { 1 }, true);
    polynomial_row_echelon_step(S6, 1, 0, { 0 }, true);

    polynomial_row_echelon_step(S6, 2, 1, { 1 }, true);
    polynomial_row_echelon_step(S6, 3, 1, { 0 }, true);

    polynomial_row_echelon_step(S6, 4, 2, { 1 }, true);
    polynomial_row_echelon_step(S6, 5, 2, { 0 }, true);

    polynomial<float, 1> hidden_variable = monomial<float, 1>{ 1, { 1 } };

    S6.row(1) = (S6.row(1) * hidden_variable) - S6.row(0);
    S6.row(3) = (S6.row(3) * hidden_variable) - S6.row(2);
    S6.row(5) = (S6.row(5) * hidden_variable) - S6.row(4);

    S6.row(1).swap(S6.row(2));
    S6.row(2).swap(S6.row(4));

    Eigen::Matrix<polynomial<float, 1>, 3, 3> S3 = S6(Eigen::seqN(3, 3), Eigen::seqN(3, 3));

    polynomial<float, 1> hidden_univariate = S3.determinant();

    std::cout << "POLY: " << hidden_univariate << std::endl;

    Eigen::Matrix<float, 1, 11> hidden_coefficients = matrix_from_polynomial_grevlex<float, 1, 11>(hidden_univariate);



    double coefficients[11];
    double z_roots[10];
    int n_roots = 0;
    for (int i = 0; i < 11; ++i) { coefficients[i] = hidden_coefficients(i); }
    if (!find_real_roots_sturm(coefficients, 10, z_roots, &n_roots, 2, 0)) { return false; }

    Eigen::Matrix<float, 10, 1> solution;
    Eigen::Matrix<float, 3, 3> fake_E;   

    for (int i = 0; i < n_roots; ++i)
    {
        float z = z_roots[i];

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> FINAL = split(substitute(S, { true }, { z }), {});

        solution = FINAL.bdcSvd(Eigen::ComputeFullV).matrixV().col(9);

        float x = solution(8) / solution(9);
        float y = solution(7) / solution(9);

        fake_E = split(substitute(E, { true, true, true }, merge<float, 2>({ x, y }, hidden_variable_index, { z })), {});

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

    return false;
}





//Eigen::Matrix<polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> ss = substitute(S, { true }, { z });//<float, 1, Eigen::Dynamic, Eigen::Dynamic>
//Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> FINAL = split(ss, {}); //<float, 1, Eigen::Dynamic, Eigen::Dynamic>
    //FINAL = matrix_from_polynomial_grevlex<float, Eigen::Dynamic, Eigen::Dynamic>(ss, 10, 10);
    //std::cout << "roots: " << std::endl;
    //std::cout << "(" << x << ", " << y << ", " << z_roots[i] << ")" << std::endl;
    //fake_E = (e(Eigen::all, 0) + x * e(Eigen::all, 1) + y * e(Eigen::all, 2) + z * e(Eigen::all, 3)).reshaped(3, 3);
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

//std::cout << "z_poly_coef" << std::endl;
//std::cout << hidden_coefficients << std::endl;
    //z_poly_coef.data()
    //polynomial<float, 1> z_poly = D(Eigen::seqN(3, 3), Eigen::seqN(3, 3)).determinant();
//std::cout << "z_poly: " << z_poly << std::endl;
    //Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> FINAL(10,10);
/*
    std::vector<int> x;
    x = { 1 };

    monomial_indices<3> aind;
    monomial_indices<3> bind;
    bind += aind;

    polynomial<float, 1> vv;
    polynomial<float, 1> vc;

    vv == vc;
    if (vv) { std::cout << "EQUAL" << std::endl; }

    monomial<float, 2> monoa = monomial<float, 2>(1.0, { 0, 1 });
    monomial<float, 2> monob = monomial<float, 2>(1.0, { 0, 1 });

    //monoa + monob;

    monomial_vector<float, 3> monov{ 1 };
    monomial_vector<float, 3> monov2{ {1, {1, 1, 1}} };

    //monov + monov2;

    monov = { monomial<float, 3>(0, { 1, 1, 1 }) };
    monov = { 5 };

    using rrtype = typename remove_polynomial<polynomial<polynomial<polynomial<float, 3>, 4>, 6>>::type;
    using rrttpl = typename remove_polynomial<polynomial<polynomial<polynomial<float, 3>, 4>, 6>>::indices;
    //rrttpl x;

    using nppt = remove_polynomial_type<float>;
    using nppi = remove_polynomial_indices<float>;
    //x.

    //std::vector<int>{ 1 } != std::vector<int>{1};
    //const bool truvec = !std::vector<int>{ 1 } != std::vector<int>{1};
    //if (truvec)
    //{

    //}
    //std::cout << std::array<float, 3>() << std::endl;

    polynomial<float, 3> test_hide = monomial_vector<float, 3>{ { 1, { 1, 1, 1}  }, { -2, { 2, 0, 1} }, {-3, {0, 2, 2}} };
    polynomial<polynomial<float, 1>, 2> hid2 = hide_in(test_hide, 2);
    polynomial<float, 3> unhid2 = unhide_in(hid2, 2);
    polynomial<polynomial<float, 2>, 1> hid1 = hide_out(test_hide, 2);
    polynomial<float, 3> unhid1 = unhide_out(hid1, 2);
    */
/*
    polynomial<float, 2> dividend = monomial_vector<float, 2>{ {2, {2, 3}}, {3, {3, 2}}, {5, {2, 1}}, {4, {1, 2}}, {10, {0,0}} };
    polynomial<float, 2> divisor = monomial_vector<float, 2>{ {6, {1, 1}}, {7, {2, 0}}, {8, {0, 2}}, {15, {0,0}} };
    monomial<float, 2> lt_dividend = leading_term<grevlex_generator<2>>(dividend);
    monomial<float, 2> lt_divisor = leading_term<grevlex_generator<2>>(divisor);
    //polynomial<float, 1> dividend = monomial_vector<float, 1>{ {5, {4}}, {6, {3}}, {7, {2}}, {8, {1}}, {10, {0}} };
    //polynomial<float, 1> divisor = monomial_vector<float, 1>{ {2, {1}}, {2, {0}} };
    //monomial<float, 1> lt_dividend = leading_term<grevlex_generator<1>>(dividend);
    //monomial<float, 1> lt_divisor = leading_term<grevlex_generator<1>>(divisor);


    std::cout << "s_polynomial: " << s_polynomial(dividend, divisor, lt_dividend, lt_divisor) << std::endl;

    for (int i = 0; i < 5; ++i)
    {
        result_reduce divres = reduce(dividend, divisor, lt_dividend, lt_divisor);
        std::cout << "REDUCE RESULT " << std::endl;
        std::cout << "irreducible:  " << divres.irreducible << std::endl;
        std::cout << "q:            " << divres.quotient << std::endl;
        std::cout << "remainder:    " << divres.remainder << std::endl;
        dividend = divres.remainder;
        lt_dividend = leading_term<grevlex_generator<2>>(dividend);
    }

    monomial_vector<float, 3> sorttest;
    grevlex_generator<3> ggsorttest;
    monomial_indices<3> prevsort;
    monomial_indices<3> nextsort;
    for (int i = 0; i < 20; ++i) {
        nextsort = ggsorttest.next().current_indices();
        sorttest.push_back({ 1, nextsort });
        //if (i > 0) { std::cout << grevlex_generator<3>::compare(prevsort, nextsort) << ", "; }
        prevsort = nextsort;
    }
    std::cout << std::endl;
    std::cout << "sorttest (def): " << sorttest << std::endl;
    sort<grevlex_generator<3>>(sorttest, false);
    std::cout << "sorttest (asc): " << sorttest << std::endl;
    sort<grevlex_generator<3>>(sorttest, true);
    std::cout << "sorttest (des): " << sorttest << std::endl;
    */

    /*
        std::cout << "HIDE TEST" << std::endl;
        std::cout << "normal:     " << test_hide << std::endl;
        std::cout << "hidden_in:  " << hid2 << std::endl;
        std::cout << "unhid_in:   " << unhid2 << std::endl;
        std::cout << "hidden_out: " << hid1 << std::endl;
        std::cout << "unhid_out:  " << unhid1 << std::endl;
        std::cout << "as_vector:  " << monomial_vector<float, 3>(test_hide) << std::endl;
        std::cout << "substitute: " << substitute(test_hide, { true, true, false }, { 2, -1, 1 }) << std::endl;
        */
        /*
        result_polynomial_division<float, 3> poly_div = polynomial_divide<grevlex_generator<3>>(E_determinant, polynomial<float, 3>(monomial_vector<float, 3>{ {1, { 0, 0, 1 }}, { 1, {0,0,2} }   }));
        std::cout << "poly_div" << std::endl;
        std::cout << "Q" << std::endl;
        poly_div.quotient.for_each([&](float const& c, monomial_indices<3> const& i) {std::cout << c << "*x^" << i[0] << "*y^" << i[1] << "*z^" << i[2] << " + "; });
        std::cout << std::endl;
        std::cout << "R" << std::endl;
        poly_div.remainder.for_each([&](float const& c, monomial_indices<3> const& i) {std::cout << c << "*x^" << i[0] << "*y^" << i[1] << "*z^" << i[2] << " + "; });
        */

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

        /*
            Eigen::Matrix<polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> S1 = substitute<float, 1, -1, -1>(S, { true }, { 0 });
            Eigen::Matrix<polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> S2 = substitute<float, 1, -1, -1>(S, { true }, { 1 });
            Eigen::Matrix<polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> S3 = substitute<float, 1, -1, -1>(S, { true }, { 2 });
            Eigen::Matrix<polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> S4 = substitute<float, 1, -1, -1>(S, { true }, { 3 });
            Eigen::Matrix<polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> S5 = substitute<float, 1, -1, -1>(S, { true }, { 4 });








            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> F1 = matrix_from_polynomial_grevlex<float, Eigen::Dynamic, Eigen::Dynamic>(S1, 100, 4);
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> F2 = matrix_from_polynomial_grevlex<float, Eigen::Dynamic, Eigen::Dynamic>(S2, 100, 4);
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> F3 = matrix_from_polynomial_grevlex<float, Eigen::Dynamic, Eigen::Dynamic>(S3, 100, 4);
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> F4 = matrix_from_polynomial_grevlex<float, Eigen::Dynamic, Eigen::Dynamic>(S4, 100, 4);
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> F5 = matrix_from_polynomial_grevlex<float, Eigen::Dynamic, Eigen::Dynamic>(S5, 100, 4);


            sizeof(std::vector<float>);
            sizeof(polynomial<float, 10>::data_type);
            sizeof(S1);
            sizeof(F1);
            sizeof(D);
            */


            //*/






                /*
                polynomial<float, 3> E_determinant = E.determinant();

                Eigen::Matrix<polynomial<float, 3>, 3, 3> EEt = E * E.transpose();
                Eigen::Matrix<polynomial<float, 3>, 3, 3> E_singular_values = (EEt * E) - ((0.5 * EEt.trace()) * E);

                Eigen::Matrix<float, 10, 20> S;

                S << matrix_from_polynomial_grevlex<float, 9, 20>(E_singular_values),
                     matrix_from_polynomial_grevlex<float, 1, 20>(E_determinant);

                Eigen::Matrix<polynomial<float, 1>, 10, 10> H;

                H << matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, 16)),
                    matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, 13)),
                    matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, 11)),
                    matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, 10)),
                    matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, { 7, 17 })),
                    matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, { 5, 14 })),
                    matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, { 4, 12 })),
                    matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, { 2,  8, 18 })),
                    matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, { 1,  6, 15 })),
                    matrix_to_polynomial_grevlex<float, 1, 10, 1>(S(Eigen::all, { 0,  3,  9, 19 }));

                for (int i = 0; i < 4; ++i) { polynomial_row_echelon_step(H, i, i, { 0 }, false); }

                Eigen::Matrix<polynomial<float, 1>, 6, 6> D = H(Eigen::seqN(4, 6), Eigen::seqN(4, 6));

                polynomial_row_echelon_step(D, 0, 0, { 1 }, true);
                polynomial_row_echelon_step(D, 1, 0, { 0 }, true);

                polynomial_row_echelon_step(D, 2, 1, { 1 }, true);
                polynomial_row_echelon_step(D, 3, 1, { 0 }, true);

                polynomial_row_echelon_step(D, 4, 2, { 1 }, true);
                polynomial_row_echelon_step(D, 5, 2, { 0 }, true);

                polynomial<float, 1> z = monomial<float, 1>{ 1, { 1 } };

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
                */


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