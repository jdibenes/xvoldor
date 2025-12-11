//
// Created by kocur on 29-Oct-24.
//

#ifndef POSELIB_THREEVIEW_PARA_H
#define POSELIB_THREEVIEW_PARA_H

#include "PoseLib/camera_pose.h"
#include <vector>
#include <Eigen/Dense>

namespace poselib {

struct ThreeViews {
    Eigen::Matrix3d R2, R3;
    Eigen::Vector3d t2, t3;

    Eigen::Matrix<double,3,4> P1() const {
        Eigen::Matrix<double,3,4> P;
        P.block<3,3>(0,0).setIdentity();
        P.block<3,1>(0,3).setZero();
        return P;
    }

    Eigen::Matrix<double,3,4> P2() const {
        Eigen::Matrix<double,3,4> P;
        P.block<3,3>(0,0) = R2;
        P.block<3,1>(0,3) = t2;
        return P;
    }

    Eigen::Matrix<double,3,4> P3() const {
        Eigen::Matrix<double,3,4> P;
        P.block<3,3>(0,0) = R3;
        P.block<3,1>(0,3) = t3;
        return P;
    }

    Eigen::Vector3d cc1() const {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d cc2() const {
        return -R2.transpose() * t2;
    }
    Eigen::Vector3d cc3() const {
        return -R3.transpose() * t3;
    }
};

typedef std::vector<ThreeViews, Eigen::aligned_allocator<ThreeViews>> ThreeViewsVector;

struct Solution : public ThreeViews {
    double T12, T23, T31;
    Eigen::Matrix<double, 1, 3> z1, z2, z3;

    Solution flipped() const {
        Solution sol = *this;
        sol.R2.col(2) *= -1.0; sol.R2.row(2) *= -1.0;
        sol.R3.col(2) *= -1.0; sol.R3.row(2) *= -1.0;
        sol.t2(2) *= -1.0; sol.t3(2) *= -1.0;
        sol.z1 *= -1.0; sol.z2 *= -1.0; sol.z3 *= -1.0;
        return sol;
    }
};

int solver_4p3v(const Eigen::Matrix<double, 2, 4> &x1,
                const Eigen::Matrix<double, 2, 4> &x2,
                const Eigen::Matrix<double, 2, 4> &x3,
                ThreeViewsVector *solutions, int iters);

double solver_4p3v_para(const Eigen::Matrix<double, 2, 4> & x1,
                        const Eigen::Matrix<double, 2, 4> & x2,
                        const Eigen::Matrix<double, 2, 4> & x3,
                        Solution *solution);

void centering_rotation(const Eigen::Vector3d& x0, Eigen::Matrix3d* R);

void solve_for_translation(const Eigen::Matrix<double, 2, 4> &x1, const Eigen::Matrix<double, 2, 4> &x2,
                           const Eigen::Matrix<double, 2, 4> &x3, const Eigen::Matrix3d &R2, const Eigen::Matrix3d &R3,
                           Eigen::Matrix<double, 3, 1> *t2, Eigen::Matrix<double, 3, 1> *t3);

void solver_4p3v_para(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                      const std::vector<Eigen::Vector2d> &x3, const std::vector<size_t> &sample,
                      std::vector<ThreeViewCameraPose> *models, int iters=100, double sq_epipolar_t = 1.0);
} // namespace poselib

#endif // POSELIB_THREEVIEW_PARA_H
