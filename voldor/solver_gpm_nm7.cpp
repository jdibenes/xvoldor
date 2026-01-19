
#include <Eigen/Eigen>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include "helpers_eigen.h"

bool solver_gpm_nm7(float const* p1, float const* p2, float* r01, float* t01)
{
    Eigen::Matrix<float, 3, 7> P1 = matrix_from_buffer<float, 3, 7>(p1);
    Eigen::Matrix<float, 3, 7> P2 = matrix_from_buffer<float, 3, 7>(p2);

    Eigen::Matrix<float, 2, 7> q1 = P1.colwise().hnormalized();
    Eigen::Matrix<float, 2, 7> q2 = P2.colwise().hnormalized();

    Eigen::Matrix<float, 3, 7> Q1 = q1.colwise().homogeneous();
    Eigen::Matrix<float, 3, 7> Q2 = q2.colwise().homogeneous();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q(7, 9);

    Q.col(0) = Q1.row(0).cwiseProduct(Q2.row(0)).transpose();
    Q.col(1) = Q1.row(1).cwiseProduct(Q2.row(0)).transpose();
    Q.col(2) = Q1.row(2).cwiseProduct(Q2.row(0)).transpose();
    Q.col(3) = Q1.row(0).cwiseProduct(Q2.row(1)).transpose();
    Q.col(4) = Q1.row(1).cwiseProduct(Q2.row(1)).transpose();
    Q.col(5) = Q1.row(2).cwiseProduct(Q2.row(1)).transpose();
    Q.col(6) = Q1.row(0).cwiseProduct(Q2.row(2)).transpose();
    Q.col(7) = Q1.row(1).cwiseProduct(Q2.row(2)).transpose();
    Q.col(8) = Q1.row(2).cwiseProduct(Q2.row(2)).transpose();

    Q.col(4) = Q.col(4) - Q.col(0);
    Q.col(8) = Q.col(8) - Q.col(0);

    Eigen::Matrix<float, 8, 1> k = Q(Eigen::all, Eigen::seqN(1, 8)).fullPivLu().kernel();
    Eigen::Matrix<float, 9, 1> e;

    e << (-(k(3) + k(7))), k;

    Eigen::Matrix<float, 3, 3> E = e.normalized().reshaped(3, 3);

    E.transposeInPlace(); // to OpenCV format

    

    cv::Mat w_E(3, 3, CV_32FC1);
    w_E.at<float>(0, 0) = E(0, 0);
    w_E.at<float>(0, 1) = E(0, 1);
    w_E.at<float>(0, 2) = E(0, 2);
    w_E.at<float>(1, 0) = E(1, 0);
    w_E.at<float>(1, 1) = E(1, 1);
    w_E.at<float>(1, 2) = E(1, 2);
    w_E.at<float>(2, 0) = E(2, 0);
    w_E.at<float>(2, 1) = E(2, 1);
    w_E.at<float>(2, 2) = E(2, 2);










    cv::Mat R1(3, 3, CV_32FC1);
    cv::Mat R2(3, 3, CV_32FC1);
    cv::Mat tx(3, 1, CV_32FC1);
    cv::decomposeEssentialMat(w_E, R1, R2, tx);

    std::cout << "DECOM" << std::endl;
    std::cout << R1 << std::endl;
    std::cout << R2 << std::endl;
    std::cout << tx << std::endl;


    /*
    std::cout << "E TEST:" << std::endl;
    for (int n = 0; n < 7; ++n)
    {
        std::cout << (Q2.col(n).transpose() * E * Q1.col(n)) << std::endl;
    }
    */



    
    

    std::vector<cv::Point2f> w_q1;
    std::vector<cv::Point2f> w_q2;
    
    for (int i = 0; i < 7; ++i)
    {
        w_q1.push_back(((cv::Point2f*)q1.data())[i]);
        w_q2.push_back(((cv::Point2f*)q2.data())[i]);
    }
    












 



    




    
    //cv::Mat w_q1(7, 2, CV_32FC1, q1.data());
    //cv::Mat w_q2(7, 2, CV_32FC1, q2.data());
    cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
    cv::Mat R(3, 3, CV_32FC1);
    cv::Mat r(3, 1, CV_32FC1, r01);
    cv::Mat t(3, 1, CV_32FC1, t01);

    cv::recoverPose(w_E, w_q1, w_q2, K, R, t);
    std::cout << "RecPose R:" << std::endl;
    std::cout << R << std::endl;

    cv::Rodrigues(R, r);

    return true;
}

/*
   
*/
/*
    std::cout << "E TEST:" << std::endl;
    std::cout << "Q1:" << std::endl;
    std::cout << Q1 << std::endl;
    std::cout << "Q2:" << std::endl;
    std::cout << Q2 << std::endl;
    std::cout << "E:" << std::endl;
    std::cout << E << std::endl;
    std::cout << E.trace() << std::endl;
    std::cout << "E TEST:" << std::endl;
    for (int n = 0; n < 7; ++n)
    {
        std::cout << (Q1.col(n).transpose() * E * Q2.col(n)) << std::endl;
    }
    */