
#include <iostream>
#include <Eigen/Eigen>
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

void planar_2_3_3(float* pa1, float* pb1, float* pa2, float* pb2, float* r01, float* t01);

int main(int argc, char* argv[])
{
    Eigen::Matrix<float, 4, 4> pose0 = load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000062.bin").transpose();
    Eigen::Matrix<float, 4, 4> pose1 = load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000072.bin").transpose();
    Eigen::Matrix<float, 4, 4> pose2 = load_pose("C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/hl2_5/pose/000082.bin").transpose();

    Eigen::Matrix<float, 4, 4> pose00h = pose0.inverse() * pose0;
    Eigen::Matrix<float, 4, 4> pose01h = pose0.inverse() * pose1;
    Eigen::Matrix<float, 4, 4> pose02h = pose0.inverse() * pose2;

    Eigen::Matrix<float, 3, 4> pose00 = pose00h(Eigen::seqN(0, 3), Eigen::all);
    Eigen::Matrix<float, 3, 4> pose01 = pose01h(Eigen::seqN(0, 3), Eigen::all);
    Eigen::Matrix<float, 3, 4> pose02 = pose02h(Eigen::seqN(0, 3), Eigen::all);

    Eigen::Matrix<float, 3, 3> R_gt = pose01(Eigen::seqN(0, 3), Eigen::seqN(0, 3));
    Eigen::AngleAxis<float> r_gt = Eigen::AngleAxis<float>(R_gt);
    Eigen::Matrix<float, 3, 1> t_gt = pose01.col(3);
    Eigen::Matrix<float, 3, 1> t_gt_planar = t_gt - (t_gt.dot(r_gt.axis()) * r_gt.axis());
    pose01.col(3) = t_gt_planar;
    std::cout << "PLANAR" << std::endl;
    std::cout << pose01.col(3).dot(r_gt.axis()) << std::endl;


    Eigen::Matrix<float, 4, 7> p1h{
        {1,   2, -3, -1.5, 4, -5, 1.5},
        {2,  -1, -2,  1.2, 3,  4,  -6},
        {10, 12, 15,    7, 9, 16,  19},
        {1,   1,  1,    1, 1,  1,   1},
    };

    //Eigen::Matrix<float, 3, 7> p1n = p1h.colwise().hnormalized();

    Eigen::Matrix<float, 3, 7> p11 = pose00 * p1h;
    Eigen::Matrix<float, 3, 7> p21 = pose01 * p1h;
    Eigen::Matrix<float, 3, 7> p31 = pose02 * p1h;

    Eigen::Matrix<float, 3, 1> r;
    Eigen::Matrix<float, 3, 1> t;


    std::cout << "INPUT" << std::endl;
    std::cout << p11 << std::endl;
    std::cout << "INPUT 2" << std::endl;
    std::cout << p21 << std::endl;


    planar_2_3_3(p11.data(), p11.data() + 3, p21.data(), p21.data() + 3, r.data(), t.data());



    std::cout << "GT" << std::endl;
    std::cout << pose01 << std::endl;
    std::cout << "POSE" << std::endl;
    std::cout << Eigen::AngleAxis<float>(r.norm(), r.normalized()).toRotationMatrix() << std::endl;
    std::cout << t << std::endl;





    //Eigen::Matrix<float, 2, 7> x11 = p11.colwise().hnormalized();
    //Eigen::Matrix<float, 2, 7> x21 = p21.colwise().hnormalized();
    //Eigen::Matrix<float, 2, 7> x31 = p31.colwise().hnormalized();







    return 0;
}
