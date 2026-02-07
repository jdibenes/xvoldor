
#define ENABLE_SOLVER_TEST

#ifdef ENABLE_SOLVER_TEST
#include <iostream>
#include <Eigen/Eigen>
#include <opencv2/calib3d.hpp>
//#include "../rolling_shutter/rnp.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"
#include "solver_gpm_hpc0.h"
#include "solver_gpm_hpc1.h"
#include "solver_gpm_hpc2.h"
#include "solver_4p3v_para.h"
#include "solver_rnp.h"
#include "trifocal.h"
#include "lock.h"

bool solver_gpm_nm7(float const* p1, float const* p2, float* r01, float* t01);
bool solver_gpm_nm6(float const* p1, float const* p2, float* r01, float* t01);
//void test_poly();


Eigen::Matrix<float, 4, 4> load_pose(char const* filename)
{
    FILE* f = fopen(filename, "rb");
    if (f == NULL) { throw std::runtime_error(""); }
    Cleaner file_close([=]() { fclose(f); });

    Eigen::Matrix<float, 4, 4> pose;
    uint32_t total = 4 * 4;
    uint32_t count = fread(pose.data(), sizeof(float), total, f);
    if (count != total) { throw std::runtime_error(""); }

    pose.transposeInPlace();

    return pose;
}

void make_planar(Eigen::Matrix<float, 3, 4>& pose)
{
    Eigen::Matrix<float, 3, 3> R_gt = pose(Eigen::seqN(0, 3), Eigen::seqN(0, 3));
    Eigen::AngleAxis<float> r_gt = Eigen::AngleAxis<float>(R_gt);
    Eigen::Matrix<float, 3, 1> t_gt = pose.col(3);
    Eigen::Matrix<float, 3, 1> t_gt_planar = t_gt - (t_gt.dot(r_gt.axis()) * r_gt.axis());
    pose.col(3) = t_gt_planar;
}



int main(int argc, char* argv[])
{
    Eigen::Matrix<float, 4, 4> pose0 = load_pose("D:/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000062.bin").transpose();
    Eigen::Matrix<float, 4, 4> pose1 = load_pose("D:/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000072.bin").transpose();
    Eigen::Matrix<float, 4, 4> pose2 = load_pose("D:/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000082.bin").transpose();

    Eigen::Matrix<float, 4, 4> pose00h = pose0.inverse() * pose0;
    Eigen::Matrix<float, 4, 4> pose01h = pose0.inverse() * pose1;
    Eigen::Matrix<float, 4, 4> pose02h = pose0.inverse() * pose2;
    Eigen::Matrix<float, 4, 4> pose12h = pose1.inverse() * pose2;

    Eigen::Matrix<float, 3, 4> pose00 = pose00h(Eigen::seqN(0, 3), Eigen::indexing::all);
    Eigen::Matrix<float, 3, 4> pose01 = pose01h(Eigen::seqN(0, 3), Eigen::indexing::all);
    Eigen::Matrix<float, 3, 4> pose02 = pose02h(Eigen::seqN(0, 3), Eigen::indexing::all);
    Eigen::Matrix<float, 3, 4> pose12 = pose12h(Eigen::seqN(0, 3), Eigen::indexing::all);
 
    make_planar(pose01);
    make_planar(pose02);
    
    //Eigen::Matrix<float, 3, 3> R_gt = pose01(Eigen::seqN(0, 3), Eigen::seqN(0, 3));
    //Eigen::AngleAxis<float> r_gt = Eigen::AngleAxis<float>(R_gt);

    //std::cout << "PLANAR" << std::endl;
    //std::cout << pose01.col(3).dot(r_gt.axis()) << std::endl;


    Eigen::Matrix<float, 4, 7> p1h{
        {1,   2, -3, -1.5, 4, -5, 1.5},
        {2,  -1, -2,  1.2, 3,  4,  -6},
        {10, 12, 15,    7, 9, 16,  19},
        {1,   1,  1,    1, 1,  1,   1},
    };

    Eigen::Matrix<float, 3, 7> p11 = pose00 * p1h;
    Eigen::Matrix<float, 3, 7> p21 = pose01 * p1h;
    Eigen::Matrix<float, 3, 7> p31 = pose02 * p1h;

    Eigen::Matrix<float, 2, 7> x11 = p11.colwise().hnormalized();
    Eigen::Matrix<float, 2, 7> x21 = p21.colwise().hnormalized();
    Eigen::Matrix<float, 2, 7> x31 = p31.colwise().hnormalized();




    Eigen::Matrix<float, 3, 3> R_gt = pose01(Eigen::seqN(0, 3), Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 1> t_gt = pose01.col(3);
    Eigen::Matrix<float, 3, 1> r;
    Eigen::Matrix<float, 3, 1> t;
    Eigen::Matrix<float, 3, 1> r2;
    Eigen::Matrix<float, 3, 1> t2;

    //solver_gpm_hpc0(p11.data(), p11.data() + 3, p31.data(), p31.data() + 3, r.data(), t.data()); // OK
    //solver_gpm_hpc1(p11.data(), p11.data() + 3, p31.data(), p31.data() + 3, r.data(), t.data(), 1); // Ok
    //solver_gpm_hpc2(p11.data(), p11.data() + 3, p11.data() + 6, p31.data(), p31.data() + 3, p31.data() + 6, r.data(), t.data(), 2); // OK
    //bool ok = trifocal_R_t_linear(x11.data(), x21.data(), x31.data(), p11.data(), 7, true, r.data(), t.data(), r2.data(), t2.data());
    //solver_4p3v_para(x11.data(), x21.data(), x31.data(), p11.data(), true, 7, r.data(), t.data(), r2.data(), t2.data());

    std::cout << "p11" << std::endl;
    std::cout << p11 << std::endl;
    std::cout << "p31" << std::endl;
    std::cout << p31 << std::endl;

    //solver_gpm_nm7(p11.data(), p31.data(), r.data(), t.data());
    solver_gpm_nm6(p11.data(), p31.data(), r.data(), t.data());

    //solver_r6p1l(p11.data(), x31.data(), 0, 0, 2, r.data(), t.data());
    //solver_r6p2l(p11.data(), x31.data(), 0, 0, r.data(), t.data());
    //solver_r6pi(p11.data(), x31.data(), 0, 0, 5, r.data(), t.data());
    /*
    RSSinglelinCameraPoseVector results1Lin;

    Eigen::MatrixXd p3D(3, 6);
    Eigen::MatrixXd p2D(2, 6);

    for (int i = 0; i < 6; ++i) {
        p3D(0, i) = p11(0, i);
        p3D(1, i) = p11(1, i);
        p3D(2, i) = p11(2, i);
        p2D(0, i) = x31(0, i);
        p2D(1, i) = x31(1, i);
    }

    R6P1Lin(p3D, p2D, 0, 0, 2, &results1Lin);
    std::cout << "R6P " << std::endl;
    for (int i = 0; i < results1Lin.size(); ++i) {
        std::cout << "SOL " << i << std::endl;
        std::cout << results1Lin[i].v << std::endl;
        std::cout << results1Lin[i].C << std::endl;
    }
    */

    Eigen::Matrix<float, 3, 3> E_gt = matrix_cross(pose02.col(3)) * pose02(Eigen::all, Eigen::seqN(0, 3));
    E_gt.normalize();




    std::cout << "GT" << std::endl;
    std::cout << "E " << std::endl;
    std::cout << E_gt << std::endl;
    
    //std::cout << pose01 << std::endl;
    //std::cout << pose12 << std::endl;
    std::cout << pose02 << std::endl;
    std::cout << "POSE" << std::endl;
    std::cout << Eigen::AngleAxis<float>(r.norm(), r.normalized()).toRotationMatrix() << std::endl;
    std::cout << t << std::endl;
    std::cout << Eigen::AngleAxis<float>(r2.norm(), r2.normalized()).toRotationMatrix() << std::endl;
    std::cout << t2 << std::endl;


    cv::Mat w_x11(7, 1, CV_32FC2, x11.data());
    cv::Mat w_x31(7, 1, CV_32FC2, x31.data());
    cv::Mat pj1 = cv::Mat::eye(3, 4, CV_32FC1);
    cv::Mat pj2(3, 4, CV_32FC1);

    pj2.at<float>(0, 0) = pose02(0, 0);
    pj2.at<float>(1, 0) = pose02(1, 0);
    pj2.at<float>(2, 0) = pose02(2, 0);

    pj2.at<float>(0, 1) = pose02(0, 1);
    pj2.at<float>(1, 1) = pose02(1, 1);
    pj2.at<float>(2, 1) = pose02(2, 1);

    pj2.at<float>(0, 2) = pose02(0, 2);
    pj2.at<float>(1, 2) = pose02(1, 2);
    pj2.at<float>(2, 2) = pose02(2, 2);

    pj2.at<float>(0, 3) = pose02(0, 3);
    pj2.at<float>(1, 3) = pose02(1, 3);
    pj2.at<float>(2, 3) = pose02(2, 3);

    cv::Mat w_X11(4, 7, CV_32FC1);
    cv::Mat w_X11i(3, 7, CV_32FC1);

    //cv::triangulatePoints(pj1, pj2, w_x11, w_x31, w_X11);
    //cv::convertPointsFromHomogeneous(w_X11, w_X11i);

    //std::cout << "TRI" << std::endl;
    //std::cout << w_X11 << std::endl;

    //test_poly();




    return 0;
}
#endif







//std::cout << "INPUT" << std::endl;
//std::cout << p11 << std::endl;
//std::cout << "INPUT 2" << std::endl;
//std::cout << p21 << std::endl;
/*
Eigen::AngleAxis<float> r12(r1.norm(), r1.normalized());
    Eigen::AngleAxis<float> r13(r2.norm(), r2.normalized());

    Eigen::Matrix<float, 3, 4> P2;
    Eigen::Matrix<float, 3, 4> P3;

    P2 << r12.toRotationMatrix(), t1;
    P3 << r13.toRotationMatrix(), t2;

    std::cout << "POSES" << std::endl;
    std::cout << pose01 << std::endl;
    std::cout << P2 << std::endl;
    std::cout << pose02 << std::endl;
    std::cout << P3 << std::endl;

    Eigen::Matrix<float, 4, 4> P2f;
    Eigen::Matrix<float, 4, 4> P3f;

    P2f << P2, Eigen::Matrix<float, 1, 4>{0, 0, 0, 1};
    P3f << P3, Eigen::Matrix<float, 1, 4>{0, 0, 0, 1};

    Eigen::Matrix<float, 4, 4> e01 = P2f.inverse() * pose01h;
    Eigen::Matrix<float, 4, 4> e02 = P3f.inverse() * pose02h;

    std::cout << "ERRORS" << std::endl;
    std::cout << e01 << std::endl;
    std::cout << e02 << std::endl;

    Eigen::Matrix<float, 3, 3> er1 = e01(Eigen::seqN(0, 3), Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 3> er2 = e02(Eigen::seqN(0, 3), Eigen::seqN(0, 3));

    Eigen::AngleAxis<float> ea1(er1);
    Eigen::AngleAxis<float> ea2(er2);

    std::cout << "rotation errors: " << (ea1.angle() * (180.0 / 3.141592653589793238463)) << " | " << (ea2.angle() * (180.0 / 3.141592653589793238463)) << std::endl;
    std::cout << "translation errors: " << (e01.col(3).hnormalized().norm()) << " | " << (e02.col(3).hnormalized().norm()) << std::endl;

    return 0;
*/


