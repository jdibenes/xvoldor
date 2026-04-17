
#define ENABLE_SOLVER_TEST

#ifdef ENABLE_SOLVER_TEST
#include <iostream>
#include <Eigen/Eigen>
#include <opencv2/calib3d.hpp>
//#include "../rolling_shutter/rnp.h"
#include <polynomial/polynomial.h>
#include "helpers_eigen.h"
#include "helpers_geometry.h"
#include "solvers.h"
#include "solver_4p3v_para.h"
#include "lock.h"

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
    Eigen::Matrix<float, 4, 4> hl2_to_opencv{
        {1,  0,  0,  0},
        {0, -1,  0,  0},
        {0,  0, -1,  0},
        {0,  0,  0,  1}
    };

    Eigen::Matrix<float, 4, 4> pose0 = hl2_to_opencv * load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000062.bin").transpose() * hl2_to_opencv;
    Eigen::Matrix<float, 4, 4> pose1 = hl2_to_opencv * load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000072.bin").transpose() * hl2_to_opencv;
    Eigen::Matrix<float, 4, 4> pose2 = hl2_to_opencv * load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000082.bin").transpose() * hl2_to_opencv;

    Eigen::Matrix<float, 4, 4> pose00h = pose0.inverse() * pose0;
    Eigen::Matrix<float, 4, 4> pose01h = pose1.inverse() * pose0;
    Eigen::Matrix<float, 4, 4> pose02h = pose2.inverse() * pose0;
    Eigen::Matrix<float, 4, 4> pose12h = pose2.inverse() * pose1;

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

    std::cout << "p21" << std::endl;
    std::cout << p21 << std::endl;
    std::cout << "p31" << std::endl;
    std::cout << p31 << std::endl;
    std::cout << "p32" << std::endl;
    std::cout << p32 << std::endl;

    Eigen::Matrix<float, 2, 7> x11 = p11.colwise().hnormalized();
    Eigen::Matrix<float, 2, 7> x21 = p21.colwise().hnormalized();
    Eigen::Matrix<float, 2, 7> x31 = p31.colwise().hnormalized();

    Eigen::Matrix<float, 3, 3> R_gt = pose01(Eigen::seqN(0, 3), Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 1> t_gt = pose01.col(3);
    Eigen::Matrix<float, 3, 3> R2_gt = pose12(Eigen::seqN(0, 3), Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 1> t2_gt = pose12.col(3);
    Eigen::Matrix<float, 3, 1> r;
    Eigen::Matrix<float, 3, 1> t;
    Eigen::Matrix<float, 3, 1> r2;
    Eigen::Matrix<float, 3, 1> t2;
    Eigen::Matrix<float, 3, 1> dr;
    Eigen::Matrix<float, 3, 1> dt;
    float focal = -1;
    bool ok;

    //ok = solver_gpm_hpc0(p11.data(), p21.data(), r.data(), t.data()); // OK*
    //ok = solver_gpm_hpc1(p11.data(), p21.data(), r.data(), t.data()); // OK*
    //ok = solver_gpm_hpc2(p11.data(), p21.data(), r.data(), t.data()); // OK*
    //ok = solver_gpm_nm5(p11.data(), p21.data(), r.data(), t.data()); // OK*
    //ok = solver_gpm_nm6(p11.data(), p21.data(), r.data(), t.data()); // OK*
    //ok = solver_gpm_nm7(p11.data(), p21.data(), r.data(), t.data()); // OK*
    //ok = solver_gpm_m4(p11.data(), p21.data(), r.data(), t.data()); // OK*
    //ok = solver_r6p1l(p11.data(), x21.data(), 0, 0, r.data(), t.data(), dr.data(), dt.data()); // OK*
    //ok = solver_r6p2i(p11.data(), x21.data(), 0, 0, r.data(), t.data(), dr.data(), dt.data(), 5); // OK*
    //ok = solver_r6p2l(p11.data(), x21.data(), 0, 0, r.data(), t.data(), dr.data(), dt.data()); // OK*
    //ok = solver_rpe_m5(p11.data(), p21.data(), r.data(), t.data()); // OK*
    //ok = solver_tft_linear(p11.data(), x21.data(), x31.data(), 7, r.data(), t.data(), r2.data(), t2.data(), 0); // OK*
    //ok = solver_4p3v_para(x11.data(), x21.data(), x31.data(), p11.data(), true, 7, r.data(), t.data(), r2.data(), t2.data());
    ok = solver_p4pf(p11.data(), x21.data(), 0, 0, r.data(), t.data(), &focal);

    std::cout << "GT" << std::endl;

    std::cout << "solver status: " << ok << std::endl;
    std::cout << pose01 << std::endl;
    std::cout << pose12 << std::endl;
    std::cout << "POSE" << std::endl;
    std::cout << Eigen::AngleAxis<float>(r.norm(), r.normalized()).toRotationMatrix() << std::endl;
    std::cout << t << std::endl;
    std::cout << Eigen::AngleAxis<float>(r2.norm(), r2.normalized()).toRotationMatrix() << std::endl;
    std::cout << t2 << std::endl;
    std::cout << "focal " << focal << std::endl;
    std::cout << "Error 01" << std::endl;
    std::cout << compute_error(vector_r_rodrigues(R_gt), t_gt, r, t) << std::endl;
    std::cout << "Error 12" << std::endl;
    std::cout << compute_error(vector_r_rodrigues(R2_gt), t2_gt, r2, t2) << std::endl;

    return 0;
}
#endif
