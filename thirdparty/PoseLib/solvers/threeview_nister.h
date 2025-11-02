//
// Created by kocur on 10-Nov-24.
//

#ifndef POSELIB_THREEVIEW_NISTER_H
#define POSELIB_THREEVIEW_NISTER_H

#include "PoseLib/camera_pose.h"
#include <vector>
#include <Eigen/Dense>

namespace poselib {
int threeview_nister(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                     const std::vector<Eigen::Vector2d> &x3, const Eigen::Vector3d &epipole,
                     const std::vector<size_t> &sample, bool use_enm, double sq_epipolar_error,
                     std::vector<ThreeViewCameraPose> *models);
}

#endif // POSELIB_THREEVIEW_NISTER_H
