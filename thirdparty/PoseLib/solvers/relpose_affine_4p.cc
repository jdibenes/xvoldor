//
// Created by kocur on 23-Oct-24.
//

#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/decompositions.h"
#include "PoseLib/misc/essential.h"
#include "PoseLib/misc/univariate.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/robust/utils.h"
#include "relpose_5pt.h"
#include "relpose_8pt.h"

#include <Eigen/Dense>
#include <iostream>

namespace poselib {

Eigen::Matrix3d relpose_affine_4p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2) {

    Eigen::Matrix<double, 4, 5> A;

    for (int i = 0; i < 4; i++) {
        A(i, 0) = x1[i](0);
        A(i, 1) = x1[i](1);
        A(i, 2) = x2[i](0);
        A(i, 3) = x2[i](1);
        A(i, 4) = 1;
    }

    Eigen::JacobiSVD<Eigen::Matrix<double, 4, 5>> svd(A, Eigen::ComputeFullV);

    Eigen::Matrix<double, 5, 1> V5 = svd.matrixV().col(4);

    Eigen::Matrix3d model;
    model << 0, 0, V5(2), 0, 0, V5(3), V5(0), V5(1), V5(4);

    return model;
}

void relpose_affine_4p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                       const std::vector<size_t> &sample, double sq_epipolar_error, bool use_enm,
                       std::vector<CameraPose> *models) {
    std::vector<Eigen::Vector2d> xx1(4), xx2(4);
    for (int i = 0; i < 4; ++i){
        xx1[i] = x1[sample[i]];
        xx2[i] = x2[sample[i]];
    }

    Eigen::Matrix3d FA = relpose_affine_4p(xx1, xx2);

    if (not use_enm){
        std::vector<Point3D> x1n(4), x2n(4);
        for (int k = 0; k < 4; ++k){
            x1n[k] = xx1[k].homogeneous().normalized();
            x2n[k] = xx2[k].homogeneous().normalized();
        }

        CameraPoseVector poses;
        motion_from_essential(FA, x1n, x2n, models);
        return;
    }

    std::vector<char> inliers;
    int num_inliers = get_inliers(FA, x1, x2, sq_epipolar_error, &inliers);

    if (num_inliers > 5){
        std::vector<Eigen::Vector3d> x1_inlier, x2_inlier;
        x1_inlier.reserve(num_inliers);
        x2_inlier.reserve(num_inliers);
        for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
            if (inliers[pt_k]) {
                x1_inlier.emplace_back(x1[pt_k].homogeneous().normalized());
                x2_inlier.emplace_back(x2[pt_k].homogeneous().normalized());
            }
        }

        relpose_5pt(x1_inlier, x2_inlier, models);
    }
}

Eigen::Matrix3d affine_homography_3pt(const std::vector<Eigen::Vector2d>& points1,
                                      const std::vector<Eigen::Vector2d>& points2) {
    Eigen::Matrix<double, 6, 6> A;
    Eigen::Matrix<double, 6, 1> b;

    // Fill matrix A and vector b
    for (int i = 0; i < 3; i++) {
        // Each point gives two equations
        int row1 = i * 2;
        int row2 = i * 2 + 1;

        // First equation: x1*a11 + y1*a12 + tx = x2
        A.row(row1) << points1[i].x(), points1[i].y(), 1, 0, 0, 0;
        b(row1) = points2[i].x();

        // Second equation: x1*a21 + y1*a22 + ty = y2
        A.row(row2) << 0, 0, 0, points1[i].x(), points1[i].y(), 1;
        b(row2) = points2[i].y();
    }

    // Solve the system
    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

    // Construct the homography matrix
    Eigen::Matrix3d H;
    H << x(0), x(1), x(2),
        x(3), x(4), x(5),
        0,    0,    1;

    return H;
}

void affine_homography_3p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                          const std::vector<size_t> &sample, double sq_epipolar_error, bool use_enm,
                          std::vector<CameraPose> *models) {
    std::vector<Eigen::Vector2d> xx1(3), xx2(3);
    for (int i = 0; i < 3; ++i){
        xx1[i] = x1[sample[i]];
        xx2[i] = x2[sample[i]];
    }

    Eigen::Matrix3d H = affine_homography_3pt(xx1, xx2);

//    //Direct H inliers
//    std::vector<char> inliers_H;
//    int num_inliers_H = get_homography_inliers(H, x1, x2, sq_epipolar_error, &inliers_H);
//
//    if (num_inliers > 5){
//        std::vector<Eigen::Vector3d> x1_inlier, x2_inlier;
//        x1_inlier.reserve(num_inliers);
//        x2_inlier.reserve(num_inliers);
//        for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
//            if (inliers[pt_k]) {
//                x1_inlier.emplace_back(x1[pt_k].homogeneous().normalized());
//                x2_inlier.emplace_back(x2[pt_k].homogeneous().normalized());
//            }
//        }
//
//        relpose_5pt(x1_inlier, x2_inlier, models);
//    }

    std::vector<Eigen::Vector3d> normals;
    std::vector<CameraPose> poses;
    motion_from_homography_svd(H, poses, normals);

    if (not use_enm) {
        for (const CameraPose& pose: poses){
            models->emplace_back(pose);
        }
        return;
    }

    for (const CameraPose& pose: poses){
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);
        std::vector<char> inliers;
        int num_inliers = get_inliers(E, x1, x2, sq_epipolar_error, &inliers);

        if (num_inliers > 5){
            std::vector<Eigen::Vector3d> x1_inlier, x2_inlier;
            x1_inlier.reserve(num_inliers);
            x2_inlier.reserve(num_inliers);
            for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
                if (inliers[pt_k]) {
                    x1_inlier.emplace_back(x1[pt_k].homogeneous().normalized());
                    x2_inlier.emplace_back(x2[pt_k].homogeneous().normalized());
                }
            }

            CameraPoseVector local_models;

            // relpose 5pt does clear
            relpose_5pt(x1_inlier, x2_inlier, &local_models);
            for (const CameraPose& local_pose : local_models){
                models->emplace_back(local_pose);
            }
        }
    }

}

void affine_essential_2p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                         const std::vector<size_t> &sample, double sq_epipolar_error, bool use_enm,
                         std::vector<CameraPose> *models) {
    std::vector<Eigen::Vector2d> xx1(2), xx2(2);
    for (int i = 0; i < 2; ++i){
        xx1[i] = x1[sample[i]];
        xx2[i] = x2[sample[i]];
    }

    Eigen::MatrixXd A(2, 4);
    A(0, 0) = xx2[0](0);
    A(0, 1) = xx2[0](1);
    A(0, 2) = xx1[0](0);
    A(0, 3) = xx1[0](1);

    A(1, 0) = xx2[1](0);
    A(1, 1) = xx2[1](1);
    A(1, 2) = xx1[1](0);
    A(1, 3) = xx1[1](1);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);

    Eigen::Matrix<double, 4, 1> V3 = svd.matrixV().col(2);
    Eigen::Matrix<double, 4, 1> V4 = svd.matrixV().col(3);

    double a = V3(0)*V3(0) + V3(1)*V3(1) - V3(2)*V3(2) - V3(3)*V3(3);
    double b = 2*(V3(0)*V4(0) + V3(1)*V4(1) - V3(2)*V4(2) - V3(3)*V4(3));
    double c = V4(0)*V4(0) + V4(1)*V4(1) - V4(2)*V4(2) - V4(3)*V4(3);

    double s[2];
    int num_sols = poselib::univariate::solve_quadratic_real(a, b, c, s);

    for (int  i = 0; i < num_sols; ++i){

        Eigen::Matrix<double, 4, 1> Ev = s[i]*V3 + V4;
        Eigen::Matrix3d E;
        E << 0, 0, Ev(0), 0, 0, Ev(1), Ev(2), Ev(3), 0;

        if (not use_enm){
            std::vector<Point3D> x1n(2), x2n(2);
            x1n[0] = xx1[0].homogeneous().normalized();
            x1n[1] = xx1[1].homogeneous().normalized();
            x2n[0] = xx2[0].homogeneous().normalized();
            x2n[1] = xx2[1].homogeneous().normalized();

            CameraPoseVector poses;
            motion_from_essential(E, x1n, x2n, models);
            return;
        }

        std::vector<char> inliers;
        int num_inliers = get_inliers(E, x1, x2, sq_epipolar_error, &inliers);

        if (num_inliers > 5){
            std::vector<Eigen::Vector3d> x1_inlier, x2_inlier;
            x1_inlier.reserve(num_inliers);
            x2_inlier.reserve(num_inliers);
            for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
                if (inliers[pt_k]) {
                    x1_inlier.emplace_back(x1[pt_k].homogeneous().normalized());
                    x2_inlier.emplace_back(x2[pt_k].homogeneous().normalized());
                }
            }
            CameraPoseVector loc_models;
            relpose_5pt(x1_inlier, x2_inlier, &loc_models);

            for (const CameraPose& pose: loc_models){
                models->emplace_back(pose);
            }
        }
    }
}

} //namespace poselib