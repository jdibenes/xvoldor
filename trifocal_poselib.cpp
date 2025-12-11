
#include "PoseLib/robust.h"

bool trifocal_R_t_poselib
(
    float const* p2d_1,
    float const* p2d_2,
    float const* p2d_3,
    float const* p2d_s,
    float const* p3d_s,
    float fx,
    float fy,
    float cx,
    float cy,
    float width,
    float height,
    int N,
    float* r1,
    float* t1,
    float* r2,
    float* t2
)
{
    std::vector<poselib::Point2D> p1;
    std::vector<poselib::Point2D> p2;
    std::vector<poselib::Point2D> p3;

    for (int i = 0; i < N; ++i)
    {
        p1.push_back(poselib::Point2D(p2d_1[2 * i], p2d_1[2 * i + 1]));
        p2.push_back(poselib::Point2D(p2d_2[2 * i], p2d_2[2 * i + 1]));
        p3.push_back(poselib::Point2D(p2d_3[2 * i], p2d_3[2 * i + 1]));
    }

    poselib::Camera cam1(1, { fx, fy, cx, cy }, width, height);
    poselib::Camera cam2(1, { fx, fy, cx, cy }, width, height);
    poselib::Camera cam3(1, { fx, fy, cx, cy }, width, height);

    poselib::RansacOptions ransac_opt;
    poselib::BundleOptions bundle_opt;

    ransac_opt.max_epipolar_error = 5.0;
    ransac_opt.progressive_sampling = false;
    ransac_opt.min_iterations = 100;
    ransac_opt.max_iterations = 1000;

    ransac_opt.lo_iterations = 0; // + nLO
    ransac_opt.inner_refine = 2; // + R
    ransac_opt.threeview_check = true; // + C
    ransac_opt.sample_sz = 5;
    ransac_opt.delta = 0.08; // 'D'
    ransac_opt.use_hc = false;
    ransac_opt.use_net = false;
    ransac_opt.init_net = false;
    ransac_opt.oracle = false;
    ransac_opt.use_affine = false; // (A)
    ransac_opt.early_lm = false; // + ELM
    ransac_opt.early_nonminimal = true; // + ENM
    ransac_opt.use_para = false; // (P)
    ransac_opt.use_nister = 0;
    ransac_opt.inner_refine_extra = 0; // 1 if + RR else 0
    ransac_opt.threeview_check_extra = false; // CC

    bundle_opt.verbose = false;
    bundle_opt.max_iterations = 100; // or 100
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_epipolar_error;

    poselib::ThreeViewCameraPose three_view_pose;
    std::vector<char> inlier_mask;

    poselib::estimate_3v_relative_pose(p1, p2, p3, cam1, cam2, cam3, ransac_opt, bundle_opt, &three_view_pose, &inlier_mask);

    Eigen::AngleAxis<double> R01(three_view_pose.pose12.R());
    Eigen::AngleAxis<double> R02(three_view_pose.pose13.R());

    Eigen::Vector3d r01 = R01.angle()* R01.axis();
    Eigen::Vector3d r02 = R02.angle()* R02.axis();

    r1[0] = r01[0];
    r1[1] = r01[1];
    r1[2] = r01[2];

    t1[0] = three_view_pose.pose12.t[0];
    t1[1] = three_view_pose.pose12.t[1];
    t1[2] = three_view_pose.pose12.t[2];

    r2[0] = r02[0];
    r2[1] = r02[1];
    r2[2] = r02[2];

    t2[0] = three_view_pose.pose13.t[0];
    t2[1] = three_view_pose.pose13.t[1];
    t2[2] = three_view_pose.pose13.t[2];

    return true;
}
