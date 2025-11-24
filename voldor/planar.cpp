
#include <iostream>
#include <Eigen/Eigen>


template <typename _scalar, int _rows, int _cols>
Eigen::Matrix<_scalar, _rows, _cols> matrix_from_buffer(_scalar const* data, int rows=_rows, int cols=_cols)
{
    Eigen::Matrix<_scalar, _rows, _cols> M(rows, cols);
    memcpy(M.data(), data, sizeof(_scalar) * rows * cols);
    return M;
}

void planar_2_3_3(float* pa1, float* pb1, float* pa2, float* pb2, float* r01, float* t01)
{
    Eigen::Matrix<float, 3, 1> PA1 = matrix_from_buffer<float, 3, 1>(pa1);
    Eigen::Matrix<float, 3, 1> PB1 = matrix_from_buffer<float, 3, 1>(pb1);
    Eigen::Matrix<float, 3, 1> PA2 = matrix_from_buffer<float, 3, 1>(pa2);
    Eigen::Matrix<float, 3, 1> PB2 = matrix_from_buffer<float, 3, 1>(pb2);

    std::cout << "SOLVER" << std::endl;
    std::cout << PA1 << std::endl;
    std::cout << PB1 << std::endl;
    std::cout << PA2 << std::endl;
    std::cout << PB2 << std::endl;


    Eigen::Matrix<float, 3, 1> V1 = (PA1 - PB1).normalized();
    Eigen::Matrix<float, 3, 1> V2 = (PA2 - PB2).normalized();

    Eigen::Matrix<float, 3, 1> kR = V1;
    float thetaR = 0;

    float thetaF = std::acos(std::clamp(V1.dot(V2), -1.0f, 1.0f));
    Eigen::Matrix<float, 3, 1> kF = (V1.cross(V2)).normalized();

    kR = kF;
    thetaR = thetaF;

    Eigen::Matrix<float, 3, 1> UA = (PA1 - PA2).normalized();
    Eigen::Matrix<float, 3, 1> UB = (PB1 - PB2).normalized();
    float LA = (PA1 - PA2).norm();
    float LB = (PB1 - PB2).norm();

    float tF_2 = thetaF / 2.0f;
    float cF_2 = std::cos(tF_2);
    float sF_2 = std::sin(tF_2);

    float ccA = -sF_2 * UA.dot(kF);
    float ccB = -sF_2 * UB.dot(kF);

    Eigen::Matrix<float, 3, 1> kG = kF.cross(V1);

    float cc;
    float cs;

    if (LA > LB)
    {
        cc = ccA;
        cs = cF_2 * UA.dot(V1) + sF_2 * UA.dot(kG);
    }
    else
    {
        cc = ccB;
        cs = cF_2 * UB.dot(V1) + sF_2 * UB.dot(kG);
    }

    Eigen::Matrix<float, 3, 1> vR = sF_2 * cs * kF + cF_2 * cc * V1 + sF_2 * cc * kG;
    float LR = vR.norm();
    kR = vR.normalized();    

    thetaR = 2 * std::acos(std::clamp((cs * cF_2) / std::sqrt(cc * cc + cs * cs), -1.0f, 1.0f));

    Eigen::Matrix<float, 3, 3> R = Eigen::AngleAxis<float>(thetaR, kR).toRotationMatrix();
    Eigen::Matrix<float, 3, 1> t = ((PA2 + PB2) - R * (PA1 + PB1)) / 2;
    Eigen::Matrix<float, 3, 1> r = kR * thetaR;

    memcpy(r01, r.data(), sizeof(float) * 3);
    memcpy(t01, t.data(), sizeof(float) * 3);
}


void planar_3_3_2_3()
{
}

void planar_4_2_2()
{

}
