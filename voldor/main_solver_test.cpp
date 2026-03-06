
#define ENABLE_SOLVER_TEST

#ifdef ENABLE_SOLVER_TEST
#include <iostream>
#include <Eigen/Eigen>
#include <opencv2/calib3d.hpp>
//#include "../rolling_shutter/rnp.h"
#include "polynomial.h"
#include "helpers_eigen.h"
#include "helpers_geometry.h"
#include "solver_gpm.h"
#include "solver_4p3v_para.h"
#include "solver_rnp.h"
#include "trifocal.h"
#include "lock.h"

bool solver_rpe_easy(float const* p1, float const* p2, float* r01, float* t01);

bool
trifocal_R_t_linear
(
    float const* p3d_1, // 3xN
    float const* p2d_2, // 2xN
    float const* p2d_3, // 2xN
    int N,
    float* r1,
    float* t1,
    float* r2,
    float* t2,
    float threshold = 0
);



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

    Eigen::Matrix<float, 4, 4> pose00h = pose0 * pose0.inverse();
    Eigen::Matrix<float, 4, 4> pose01h = pose1 * pose0.inverse();
    Eigen::Matrix<float, 4, 4> pose02h = pose2 * pose0.inverse();
    Eigen::Matrix<float, 4, 4> pose12h = pose2 * pose1.inverse();

    Eigen::Matrix<float, 3, 4> pose00 = pose00h(Eigen::seqN(0, 3), Eigen::indexing::all);
    Eigen::Matrix<float, 3, 4> pose01 = pose01h(Eigen::seqN(0, 3), Eigen::indexing::all);
    Eigen::Matrix<float, 3, 4> pose02 = pose02h(Eigen::seqN(0, 3), Eigen::indexing::all);
    Eigen::Matrix<float, 3, 4> pose12 = pose12h(Eigen::seqN(0, 3), Eigen::indexing::all);
 
    //make_planar(pose01);
    //make_planar(pose02);

    Eigen::Matrix<float, 4, 7> p1h{
        {1,   2, -3, -1.5, 4, -5, 1.5},
        {2,  -1, -2,  1.2, 3,  4,  -6},
        {10, 12, 15,    7, 9, 16,  19},
        {1,   1,  1,    1, 1,  1,   1},
    };

    Eigen::Matrix<float, 3, 7> p11 = pose00 * p1h;
    Eigen::Matrix<float, 3, 7> p21 = pose01 * p1h;
    Eigen::Matrix<float, 3, 7> p31 = pose02 * p1h;
    Eigen::Matrix<float, 3, 7> p32 = pose12 * p21.colwise().homogeneous();

    std::cout << "p31" << std::endl;
    std::cout << p31 << std::endl;
    std::cout << "p32" << std::endl;
    std::cout << p32 << std::endl;

    Eigen::Matrix<float, 2, 7> x11 = p11.colwise().hnormalized();
    Eigen::Matrix<float, 2, 7> x21 = p21.colwise().hnormalized();
    Eigen::Matrix<float, 2, 7> x31 = p31.colwise().hnormalized();

    Eigen::Matrix<float, 3, 3> R_gt = pose01(Eigen::seqN(0, 3), Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 1> t_gt = pose01.col(3);
    Eigen::Matrix<float, 3, 1> r;
    Eigen::Matrix<float, 3, 1> t;
    Eigen::Matrix<float, 3, 1> r2;
    Eigen::Matrix<float, 3, 1> t2;
    bool ok;

    //ok = solver_gpm_hpc0(p11.data(), p11.data() + 3, p31.data(), p31.data() + 3, r.data(), t.data()); // OK
    //ok = solver_gpm_hpc1(p11.data(), p11.data() + 3, p31.data(), p31.data() + 3, r.data(), t.data(), 1); // Ok
    //ok = solver_gpm_hpc2(p11.data(), p11.data() + 3, p11.data() + 6, p31.data(), p31.data() + 3, p31.data() + 6, r.data(), t.data(), 2); // OK
    //ok = trifocal_R_t_linear(x11.data(), x21.data(), x31.data(), p11.data(), 7, true, r.data(), t.data(), r2.data(), t2.data());
    ok = trifocal_R_t_linear(p11.data(), x21.data(), x31.data(), 7, r.data(), t.data(), r2.data(), t2.data());
    //ok = solver_4p3v_para(x11.data(), x21.data(), x31.data(), p11.data(), true, 7, r.data(), t.data(), r2.data(), t2.data());
    // 
    //ok = solver_rpe_easy(p11.data(), p21.data(), r.data(), t.data());

    //ok = solver_gpm_nm7(p11.data(), p31.data(), r.data(), t.data());
    //ok = solver_gpm_nm6(p11.data(), p31.data(), r.data(), t.data());
    //ok = solver_gpm_nm5(p11.data(), p21.data(), r.data(), t.data());

    //ok = solver_r6p1l(p11.data(), x31.data(), 0, 0, 2, r.data(), t.data());
    //ok = solver_r6p2l(p11.data(), x31.data(), 0, 0, r.data(), t.data());
    //ok = solver_r6pi(p11.data(), x31.data(), 0, 0, 5, r.data(), t.data());
    
    std::cout << "GT" << std::endl;

    std::cout << "solver status: " << ok << std::endl;
    std::cout << pose01 << std::endl;
    std::cout << pose12 << std::endl;
    std::cout << pose02 << std::endl;
    std::cout << "POSE" << std::endl;
    std::cout << Eigen::AngleAxis<float>(r.norm(), r.normalized()).toRotationMatrix() << std::endl;
    std::cout << t << std::endl;
    std::cout << Eigen::AngleAxis<float>(r2.norm(), r2.normalized()).toRotationMatrix() << std::endl;
    std::cout << t2 << std::endl;

    return 0;
}
#endif
