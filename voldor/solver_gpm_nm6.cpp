
#include <Eigen/Eigen>
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




class poly
{
    void* root;
    int variables;


    poly(int variables) 
    {
        this->variables = variables;
        this->root = nullptr;
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


    poly operator+(poly& other)
    {



        //for (int i = 0; )


    }

    poly operator-(poly const& other)
    {

    }

    poly operator*(poly const& other)
    {

    }



};






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
