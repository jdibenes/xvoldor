//
// Created by kocur on 23-Oct-24.
//

#ifndef POSELIB_RELPOSE_AFFINE_4P_H
#define POSELIB_RELPOSE_AFFINE_4P_H

#include "PoseLib/camera_pose.h"
#include <Eigen/Dense>
#include <vector>

namespace poselib {
Eigen::Matrix3d relpose_affine_4p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2);
void relpose_affine_4p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                       const std::vector<size_t> &sample, double sq_epipolar_error, bool use_enm,
                       std::vector<CameraPose> *models);

void affine_homography_3p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                          const std::vector<size_t> &sample, double sq_epipolar_error, bool use_enm,
                          std::vector<CameraPose> *models);

void affine_essential_2p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                         const std::vector<size_t> &sample, double sq_epipolar_error, bool use_enm,
                         std::vector<CameraPose> *models);
}

#endif // POSELIB_RELPOSE_AFFINE_4P_H
