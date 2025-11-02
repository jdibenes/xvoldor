// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_JACOBIAN_IMPL_H_
#define POSELIB_JACOBIAN_IMPL_H_

#include <iostream>
#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/colmap_models.h"
#include "PoseLib/misc/essential.h"
#include "PoseLib/types.h"
#include "utils.h"

namespace poselib {

// For the accumulators we support supplying a vector<double> with point-wise weights for the residuals
// In case we don't want to have weighted residuals, we can pass UniformWeightVector instead of filling a std::vector
// with 1.0 The multiplication is then hopefully is optimized away since it always returns 1.0
class UniformWeightVector {
  public:
    UniformWeightVector() {}
    constexpr double operator[](std::size_t idx) const { return 1.0; }
};
class UniformWeightVectors { // this corresponds to std::vector<std::vector<double>> used for generalized cameras etc
  public:
    UniformWeightVectors() {}
    constexpr const UniformWeightVector &operator[](std::size_t idx) const { return w; }
    const UniformWeightVector w;
    typedef UniformWeightVector value_type;
};

template <typename CameraModel, typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class CameraJacobianAccumulator {
  public:
    CameraJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                              const Camera &cam, const LossFunction &loss,
                              const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D), X(points3D), camera(cam), loss_fn(loss), weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = pose.apply(X[i]);
            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (Z(2) < 0)
                continue;
            const double inv_z = 1.0 / Z(2);
            Eigen::Vector2d p(Z(0) * inv_z, Z(1) * inv_z);
            CameraModel::project(camera.params, p, &p);
            const double r0 = p(0) - x[i](0);
            const double r1 = p(1) - x[i](1);
            const double r_squared = r0 * r0 + r1 * r1;
            cost += weights[i] * loss_fn.loss(r_squared);
        }
        return cost;
    }

    // computes J.transpose() * J and J.transpose() * res
    // Only computes the lower half of JtJ
    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        Eigen::Matrix3d R = pose.R();
        Eigen::Matrix2d Jcam;
        Jcam.setIdentity(); // we initialize to identity here (this is for the calibrated case)
        size_t num_residuals = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = R * X[i] + pose.t;
            const Eigen::Vector2d z = Z.hnormalized();

            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (Z(2) < 0)
                continue;

            // Project with intrinsics
            Eigen::Vector2d zp = z;
            CameraModel::project_with_jac(camera.params, z, &zp, &Jcam);

            // Setup residual
            Eigen::Vector2d r = zp - x[i];
            const double r_squared = r.squaredNorm();
            const double weight = weights[i] * loss_fn.weight(r_squared);

            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute jacobian w.r.t. Z (times R)
            Eigen::Matrix<double, 2, 3> dZ;
            dZ.block<2, 2>(0, 0) = Jcam;
            dZ.col(2) = -Jcam * z;
            dZ *= 1.0 / Z(2);
            dZ *= R;

            const double X0 = X[i](0);
            const double X1 = X[i](1);
            const double X2 = X[i](2);
            const double dZtdZ_0_0 = weight * dZ.col(0).dot(dZ.col(0));
            const double dZtdZ_1_0 = weight * dZ.col(1).dot(dZ.col(0));
            const double dZtdZ_1_1 = weight * dZ.col(1).dot(dZ.col(1));
            const double dZtdZ_2_0 = weight * dZ.col(2).dot(dZ.col(0));
            const double dZtdZ_2_1 = weight * dZ.col(2).dot(dZ.col(1));
            const double dZtdZ_2_2 = weight * dZ.col(2).dot(dZ.col(2));
            JtJ(0, 0) += X2 * (X2 * dZtdZ_1_1 - X1 * dZtdZ_2_1) + X1 * (X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1);
            JtJ(1, 0) += -X2 * (X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1) - X1 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
            JtJ(2, 0) += X1 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0) - X2 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
            JtJ(3, 0) += X1 * dZtdZ_2_0 - X2 * dZtdZ_1_0;
            JtJ(4, 0) += X1 * dZtdZ_2_1 - X2 * dZtdZ_1_1;
            JtJ(5, 0) += X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1;
            JtJ(1, 1) += X2 * (X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0) + X0 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
            JtJ(2, 1) += -X2 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) - X0 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0);
            JtJ(3, 1) += X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0;
            JtJ(4, 1) += X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1;
            JtJ(5, 1) += X2 * dZtdZ_2_0 - X0 * dZtdZ_2_2;
            JtJ(2, 2) += X1 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) + X0 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
            JtJ(3, 2) += X0 * dZtdZ_1_0 - X1 * dZtdZ_0_0;
            JtJ(4, 2) += X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0;
            JtJ(5, 2) += X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0;
            JtJ(3, 3) += dZtdZ_0_0;
            JtJ(4, 3) += dZtdZ_1_0;
            JtJ(5, 3) += dZtdZ_2_0;
            JtJ(4, 4) += dZtdZ_1_1;
            JtJ(5, 4) += dZtdZ_2_1;
            JtJ(5, 5) += dZtdZ_2_2;
            r *= weight;
            Jtr(0) += (r(0) * (X1 * dZ(0, 2) - X2 * dZ(0, 1)) + r(1) * (X1 * dZ(1, 2) - X2 * dZ(1, 1)));
            Jtr(1) += (-r(0) * (X0 * dZ(0, 2) - X2 * dZ(0, 0)) - r(1) * (X0 * dZ(1, 2) - X2 * dZ(1, 0)));
            Jtr(2) += (r(0) * (X0 * dZ(0, 1) - X1 * dZ(0, 0)) + r(1) * (X0 * dZ(1, 1) - X1 * dZ(1, 0)));
            Jtr(3) += (dZ(0, 0) * r(0) + dZ(1, 0) * r(1));
            Jtr(4) += (dZ(0, 1) * r(0) + dZ(1, 1) * r(1));
            Jtr(5) += (dZ(0, 2) * r(0) + dZ(1, 2) * r(1));
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        // The rotation is parameterized via the lie-rep. and post-multiplication
        //   i.e. R(delta) = R * expm([delta]_x)
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));

        // Translation is parameterized as (negative) shift in position
        //  i.e. t(delta) = t + R*delta
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    const Camera &camera;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

template <typename LossFunction, typename ResidualWeightVectors = UniformWeightVectors>
class GeneralizedCameraJacobianAccumulator {
  public:
    GeneralizedCameraJacobianAccumulator(const std::vector<std::vector<Point2D>> &points2D,
                                         const std::vector<std::vector<Point3D>> &points3D,
                                         const std::vector<CameraPose> &camera_ext,
                                         const std::vector<Camera> &camera_int, const LossFunction &l,
                                         const ResidualWeightVectors &w = ResidualWeightVectors())
        : num_cams(points2D.size()), x(points2D), X(points3D), rig_poses(camera_ext), cameras(camera_int), loss_fn(l),
          weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0.0;
        for (size_t k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            const Camera &camera = cameras[k];
            CameraPose full_pose;
            full_pose.q = quat_multiply(rig_poses[k].q, pose.q);
            full_pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;

            switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id: {                                                                                            \
        CameraJacobianAccumulator<Model, decltype(loss_fn), typename ResidualWeightVectors::value_type> accum(         \
            x[k], X[k], cameras[k], loss_fn, weights[k]);                                                              \
        cost += accum.residual(full_pose);                                                                             \
        break;                                                                                                         \
    }
                SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
            }
        }
        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        size_t num_residuals = 0;

        for (size_t k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            const Camera &camera = cameras[k];
            CameraPose full_pose;
            full_pose.q = quat_multiply(rig_poses[k].q, pose.q);
            full_pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;

            switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id: {                                                                                            \
        CameraJacobianAccumulator<Model, decltype(loss_fn), typename ResidualWeightVectors::value_type> accum(         \
            x[k], X[k], cameras[k], loss_fn, weights[k]);                                                              \
        num_residuals += accum.accumulate(full_pose, JtJ, Jtr);                                                        \
        break;                                                                                                         \
    }
                SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const size_t num_cams;
    const std::vector<std::vector<Point2D>> &x;
    const std::vector<std::vector<Point3D>> &X;
    const std::vector<CameraPose> &rig_poses;
    const std::vector<Camera> &cameras;
    const LossFunction &loss_fn;
    const ResidualWeightVectors &weights;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector> class LineJacobianAccumulator {
  public:
    LineJacobianAccumulator(const std::vector<Line2D> &lines2D_, const std::vector<Line3D> &lines3D_,
                            const LossFunction &loss, const ResidualWeightVector &w = ResidualWeightVector())
        : lines2D(lines2D_), lines3D(lines3D_), loss_fn(loss), weights(w) {}

    double residual(const CameraPose &pose) const {
        Eigen::Matrix3d R = pose.R();
        double cost = 0;
        for (size_t i = 0; i < lines2D.size(); ++i) {
            const Eigen::Vector3d Z1 = R * lines3D[i].X1 + pose.t;
            const Eigen::Vector3d Z2 = R * lines3D[i].X2 + pose.t;
            Eigen::Vector3d l = Z1.cross(Z2);
            l /= l.topRows<2>().norm();

            const double r0 = l.dot(lines2D[i].x1.homogeneous());
            const double r1 = l.dot(lines2D[i].x2.homogeneous());
            const double r_squared = r0 * r0 + r1 * r1;
            cost += weights[i] * loss_fn.loss(r_squared);
        }
        return cost;
    }

    // computes J.transpose() * J and J.transpose() * res
    // Only computes the lower half of JtJ
    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {

        Eigen::Matrix3d E, R;
        R = pose.R();
        E << pose.t.cross(R.col(0)), pose.t.cross(R.col(1)), pose.t.cross(R.col(2));
        size_t num_residuals = 0;
        for (size_t k = 0; k < lines2D.size(); ++k) {
            const Eigen::Vector3d Z1 = R * lines3D[k].X1 + pose.t;
            const Eigen::Vector3d Z2 = R * lines3D[k].X2 + pose.t;

            const Eigen::Vector3d X12 = lines3D[k].X1.cross(lines3D[k].X2);
            const Eigen::Vector3d dX = lines3D[k].X1 - lines3D[k].X2;

            // Projected line
            const Eigen::Vector3d l = Z1.cross(Z2);

            // Normalized line by first two coordinates
            Eigen::Vector2d alpha = l.topRows<2>();
            double beta = l(2);
            const double n_alpha = alpha.norm();
            alpha /= n_alpha;
            beta /= n_alpha;

            // Compute residual
            Eigen::Vector2d r;
            r << alpha.dot(lines2D[k].x1) + beta, alpha.dot(lines2D[k].x2) + beta;

            const double r_squared = r.squaredNorm();
            const double weight = weights[k] * loss_fn.weight(r_squared);

            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            Eigen::Matrix<double, 3, 6> dl_drt;
            // Differentiate line with respect to rotation parameters
            dl_drt.block<1, 3>(0, 0) = E.row(0).cross(dX) - R.row(0).cross(X12);
            dl_drt.block<1, 3>(1, 0) = E.row(1).cross(dX) - R.row(1).cross(X12);
            dl_drt.block<1, 3>(2, 0) = E.row(2).cross(dX) - R.row(2).cross(X12);
            // and translation params
            dl_drt.block<1, 3>(0, 3) = R.row(0).cross(dX);
            dl_drt.block<1, 3>(1, 3) = R.row(1).cross(dX);
            dl_drt.block<1, 3>(2, 3) = R.row(2).cross(dX);

            // Differentiate normalized line w.r.t. original line
            Eigen::Matrix3d dln_dl;
            dln_dl.block<2, 2>(0, 0) = (Eigen::Matrix2d::Identity() - alpha * alpha.transpose()) / n_alpha;
            dln_dl.block<1, 2>(2, 0) = -beta * alpha / n_alpha;
            dln_dl.block<2, 1>(0, 2).setZero();
            dln_dl(2, 2) = 1 / n_alpha;

            // Differentiate residual w.r.t. line
            Eigen::Matrix<double, 2, 3> dr_dl;
            dr_dl.row(0) << lines2D[k].x1.transpose(), 1.0;
            dr_dl.row(1) << lines2D[k].x2.transpose(), 1.0;

            Eigen::Matrix<double, 2, 6> J = dr_dl * dln_dl * dl_drt;
            // Accumulate into JtJ and Jtr
            Jtr += weight * J.transpose() * r;
            for (size_t i = 0; i < 6; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J.col(i).dot(J.col(j)));
                }
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        // The rotation is parameterized via the lie-rep. and post-multiplication
        //   i.e. R(delta) = R * expm([delta]_x)
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        // Translation is parameterized as (negative) shift in position
        //  i.e. t(delta) = t + R*delta
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<Line2D> &lines2D;
    const std::vector<Line3D> &lines3D;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

template <typename PointLossFunction, typename LineLossFunction, typename PointResidualsVector = UniformWeightVector,
          typename LineResidualsVector = UniformWeightVector>
class PointLineJacobianAccumulator {
  public:
    PointLineJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                 const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                                 const PointLossFunction &l_point, const LineLossFunction &l_line,
                                 const PointResidualsVector &weights_pts = PointResidualsVector(),
                                 const LineResidualsVector &weights_l = LineResidualsVector())
        : pts_accum(points2D, points3D, trivial_camera, l_point, weights_pts),
          line_accum(lines2D, lines3D, l_line, weights_l) {
        trivial_camera.model_id = NullCameraModel::model_id;
    }

    double residual(const CameraPose &pose) const { return pts_accum.residual(pose) + line_accum.residual(pose); }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        return pts_accum.accumulate(pose, JtJ, Jtr) + line_accum.accumulate(pose, JtJ, Jtr);
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        // Both CameraJacobianAccumulator and LineJacobianAccumulator have the same step!
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    Camera trivial_camera;
    CameraJacobianAccumulator<NullCameraModel, PointLossFunction, PointResidualsVector> pts_accum;
    LineJacobianAccumulator<LineLossFunction, LineResidualsVector> line_accum;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class RelativePoseJacobianAccumulator {
  public:
    RelativePoseJacobianAccumulator(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                    const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const CameraPose &pose) const {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());
            double nJc_sq = (E.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 5, 5> &JtJ, Eigen::Matrix<double, 5, 1> &Jtr) {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(pose.t.x()) < std::abs(pose.t.y())) {
            // x < y
            if (std::abs(pose.t.x()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(pose.t.y()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(pose.t).normalized();

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(tangent_basis[k])*R)
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), E.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r * r);
            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 5> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            JtJ(0, 0) += weight * (J(0) * J(0));
            JtJ(1, 0) += weight * (J(1) * J(0));
            JtJ(1, 1) += weight * (J(1) * J(1));
            JtJ(2, 0) += weight * (J(2) * J(0));
            JtJ(2, 1) += weight * (J(2) * J(1));
            JtJ(2, 2) += weight * (J(2) * J(2));
            JtJ(3, 0) += weight * (J(3) * J(0));
            JtJ(3, 1) += weight * (J(3) * J(1));
            JtJ(3, 2) += weight * (J(3) * J(2));
            JtJ(3, 3) += weight * (J(3) * J(3));
            JtJ(4, 0) += weight * (J(4) * J(0));
            JtJ(4, 1) += weight * (J(4) * J(1));
            JtJ(4, 2) += weight * (J(4) * J(2));
            JtJ(4, 3) += weight * (J(4) * J(3));
            JtJ(4, 4) += weight * (J(4) * J(4));
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 5, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + tangent_basis * dp.block<2, 1>(3, 0);
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 5;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class ThreeViewRelativePoseJacobianAccumulator {
  public:
    ThreeViewRelativePoseJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                             const std::vector<Point2D> &points2D_2,
                                             const std::vector<Point2D> &points2D_3,
                                             const LossFunction &l,
                                             const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), x3(points2D_3), loss_fn(l), weights(w) {}

    double residual(const ThreeViewCameraPose &three_view_pose) const {
        Eigen::Matrix3d E12, E13, E23;
        essential_from_motion(three_view_pose.pose12, &E12);
        essential_from_motion(three_view_pose.pose13, &E13);
        essential_from_motion(three_view_pose.pose23(), &E23);

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            // E12          
            double C12 = x2[k].homogeneous().dot(E12 * x1[k].homogeneous());
            double nJc12_sq = (E12.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (E12.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();
            double r12_sq = (C12 * C12) / nJc12_sq;
            double loss12 = loss_fn.loss(r12_sq);
            if (loss12 == 0.0)
                continue;

            // E13            
            double C13 = x3[k].homogeneous().dot(E13 * x1[k].homogeneous());
            double nJc13_sq = (E13.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                              (E13.block<3, 2>(0, 0).transpose() * x3[k].homogeneous()).squaredNorm();
            
            double r13_sq = (C13 * C13) / nJc13_sq;
            double loss13 = loss_fn.loss(r13_sq);
            if (loss13 == 0.0)
                continue;
            
            // E23            
            double C23 = x3[k].homogeneous().dot(E23 * x2[k].homogeneous());
            double nJc23_sq = (E23.block<2, 3>(0, 0) * x2[k].homogeneous()).squaredNorm() +
                              (E23.block<3, 2>(0, 0).transpose() * x3[k].homogeneous()).squaredNorm();

            double r23_sq = (C23 * C23) / nJc23_sq;
            double loss23 = loss_fn.loss(r23_sq);
            if (loss23 == 0.0)
                continue;

            cost += weights[k] * loss12;
            cost += weights[k] * loss13;
            cost += weights[k] * loss23;
        }

        return cost;
    }

    size_t accumulate(const ThreeViewCameraPose &three_view_pose, Eigen::Matrix<double, 11, 11> &JtJ,
                      Eigen::Matrix<double, 11, 1> &Jtr) {
        // We use tangent bases for t12 and direct vector for t23
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        
        CameraPose pose12 = three_view_pose.pose12;
        
        if (std::abs(pose12.t.x()) < std::abs(pose12.t.y())) {
            // x < y
            if (std::abs(pose12.t.x()) < std::abs(pose12.t.z())) {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(pose12.t.y()) < std::abs(pose12.t.z())) {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(pose12.t).normalized();

        Eigen::Matrix3d E12, R12, E13, R13, E23, R23;

        R12 = pose12.R();
        essential_from_motion(pose12, &E12);
        
        R13 = three_view_pose.pose13.R();
        essential_from_motion(three_view_pose.pose13, &E13);

        CameraPose pose23 = three_view_pose.pose23();
        R23 = pose23.R();
        essential_from_motion(pose23, &E23);
                
        // Matrices contain the jacobians of E12 w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dE12dr12;
        Eigen::Matrix<double, 9, 2> dE12dt12;

        // Each column is vec(E12*skew(e_k)) where e_k is k:th basis vector
        dE12dr12.block<3, 1>(0, 0).setZero();
        dE12dr12.block<3, 1>(0, 1) = -E12.col(2);
        dE12dr12.block<3, 1>(0, 2) = E12.col(1);
        dE12dr12.block<3, 1>(3, 0) = E12.col(2);
        dE12dr12.block<3, 1>(3, 1).setZero();
        dE12dr12.block<3, 1>(3, 2) = -E12.col(0);
        dE12dr12.block<3, 1>(6, 0) = -E12.col(1);
        dE12dr12.block<3, 1>(6, 1) = E12.col(0);
        dE12dr12.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(tangent_basis[k])*R12)
        dE12dt12.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R12.col(0));
        dE12dt12.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R12.col(0));
        dE12dt12.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R12.col(1));
        dE12dt12.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R12.col(1));
        dE12dt12.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R12.col(2));
        dE12dt12.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R12.col(2));
        
        // Matrices contain the jacobians of E12 w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dE13dr13;
        Eigen::Matrix<double, 9, 3> dE13dt13;
        
        // Each column is vec(E13*skew(e_k)) where e_k is k:th basis vector
        dE13dr13.block<3, 1>(0, 0).setZero();
        dE13dr13.block<3, 1>(0, 1) = -E13.col(2);
        dE13dr13.block<3, 1>(0, 2) = E13.col(1);
        dE13dr13.block<3, 1>(3, 0) = E13.col(2);
        dE13dr13.block<3, 1>(3, 1).setZero();
        dE13dr13.block<3, 1>(3, 2) = -E13.col(0);
        dE13dr13.block<3, 1>(6, 0) = -E13.col(1);
        dE13dr13.block<3, 1>(6, 1) = E13.col(0);
        dE13dr13.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(e_k)*R13) where e_k is k:th basis vector
        Eigen::Matrix3d dE13dt13_0, dE13dt13_1, dE13dt13_2;
        dE13dt13_0.row(0).setZero();
        dE13dt13_0.row(1) = -R13.row(2);
        dE13dt13_0.row(2) = R13.row(1);

        dE13dt13_1.row(0) = R13.row(2);
        dE13dt13_1.row(1).setZero();
        dE13dt13_1.row(2) = - R13.row(0);

        dE13dt13_2.row(0) = - R13.row(1);
        dE13dt13_2.row(1) = R13.row(0);
        dE13dt13_2.row(2).setZero();

        dE13dt13.col(0) = Eigen::Map<Eigen::VectorXd>(dE13dt13_0.data(), dE13dt13_0.size());
        dE13dt13.col(1) = Eigen::Map<Eigen::VectorXd>(dE13dt13_1.data(), dE13dt13_1.size());
        dE13dt13.col(2) = Eigen::Map<Eigen::VectorXd>(dE13dt13_2.data(), dE13dt13_2.size());

        //TODO: this part calculates dE23dX and is not optimized yet

        // define skew(e_k)
        Eigen::Matrix3d b_0 = skew(Eigen::Vector3d::UnitX());
        Eigen::Matrix3d b_1 = skew(Eigen::Vector3d::UnitY());
        Eigen::Matrix3d b_2 = skew(Eigen::Vector3d::UnitZ());

        Eigen::Matrix3d dE23dr12_0, dE23dr12_1, dE23dr12_2;
        dE23dr12_0 = skew(R13 * b_0 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_0 * R12.transpose();
        dE23dr12_1 = skew(R13 * b_1 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_1 * R12.transpose();
        dE23dr12_2 = skew(R13 * b_2 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_2 * R12.transpose();

        Eigen::Matrix3d dE23dr13_0, dE23dr13_1, dE23dr13_2;
        dE23dr13_0 = - dE23dr12_0;
        dE23dr13_1 = - dE23dr12_1;
        dE23dr13_2 = - dE23dr12_2;

        // dE23dt12 = skew(tangent_basis_k) * R23
        Eigen::Matrix3d dE23dt12_0, dE23dt12_1;
        dE23dt12_0 = - skew(R23 * tangent_basis.col(0)) * R23;
        dE23dt12_1 = - skew(R23 * tangent_basis.col(1)) * R23;

        // dE23dt13 = skew(e_k) * R23
        Eigen::Matrix3d dE23dt13_0, dE23dt13_1, dE23dt13_2;
        dE23dt13_0.row(0).setZero();
        dE23dt13_0.row(1) = -R23.row(2);
        dE23dt13_0.row(2) = R23.row(1);

        dE23dt13_1.row(0) = R23.row(2);
        dE23dt13_1.row(1).setZero();
        dE23dt13_1.row(2) = - R23.row(0);

        dE23dt13_2.row(0) = - R23.row(1);
        dE23dt13_2.row(1) = R23.row(0);
        dE23dt13_2.row(2).setZero();

        Eigen::Matrix<double, 9, 3> dE23dr12;
        Eigen::Matrix<double, 9, 3> dE23dr13;
        Eigen::Matrix<double, 9, 2> dE23dt12;
        Eigen::Matrix<double, 9, 3> dE23dt13;

        dE23dr12.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dr12_0.data(), dE23dr12_0.size());
        dE23dr12.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dr12_1.data(), dE23dr12_1.size());
        dE23dr12.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dr12_2.data(), dE23dr12_2.size());

        dE23dr13.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dr13_0.data(), dE23dr13_0.size());
        dE23dr13.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dr13_1.data(), dE23dr13_1.size());
        dE23dr13.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dr13_2.data(), dE23dr13_2.size());

        dE23dt12.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dt12_0.data(), dE23dt12_0.size());
        dE23dt12.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dt12_1.data(), dE23dt12_1.size());

        dE23dt13.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dt13_0.data(), dE23dt13_0.size());
        dE23dt13.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dt13_1.data(), dE23dt13_1.size());
        dE23dt13.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dt13_2.data(), dE23dt13_2.size());


        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C12 = x2[k].homogeneous().dot(E12 * x1[k].homogeneous());
            double C13 = x3[k].homogeneous().dot(E13 * x1[k].homogeneous());
            double C23 = x3[k].homogeneous().dot(E23 * x2[k].homogeneous());

            // J_C12 is the Jacobian of the epipolar constraint w.r12.t. the image points
            Eigen::Vector4d J_C12;
            J_C12 << E12.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), E12.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C12 = J_C12.norm();
            const double inv_nJ_C12 = 1.0 / nJ_C12;
            const double r12 = C12 * inv_nJ_C12;
            const double weight12 = weights[k] * loss_fn.weight(r12 * r12);
            if (weight12 == 0.0) {
                continue;
            }
                        
            Eigen::Vector4d J_C13;
            J_C13 << E13.block<3, 2>(0, 0).transpose() * x3[k].homogeneous(), E13.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C13 = J_C13.norm();
            const double inv_nJ_C13 = 1.0 / nJ_C13;
            const double r13 = C13 * inv_nJ_C13;

            const double weight13 = weights[k] * loss_fn.weight(r13 * r13);
            if (weight13 == 0.0) {
                continue;
            }
            
            Eigen::Vector4d J_C23;
            J_C23 << E23.block<3, 2>(0, 0).transpose() * x3[k].homogeneous(), E23.block<2, 3>(0, 0) * x2[k].homogeneous();
            const double nJ_C23 = J_C23.norm();
            const double inv_nJ_C23 = 1.0 / nJ_C23;
            const double r23 = C23 * inv_nJ_C23;

            const double weight23 = weights[k] * loss_fn.weight(r23 * r23);
            if (weight23 == 0.0) {
                continue;
            }
                        
            num_residuals += 3;

            // Compute Jacobian of Sampson error w.r12.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dSdE12;
            dSdE12 << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s12 = C12 * inv_nJ_C12 * inv_nJ_C12;
            dSdE12(0) -= s12 * (J_C12(2) * x1[k](0) + J_C12(0) * x2[k](0));
            dSdE12(1) -= s12 * (J_C12(3) * x1[k](0) + J_C12(0) * x2[k](1));
            dSdE12(2) -= s12 * (J_C12(0));
            dSdE12(3) -= s12 * (J_C12(2) * x1[k](1) + J_C12(1) * x2[k](0));
            dSdE12(4) -= s12 * (J_C12(3) * x1[k](1) + J_C12(1) * x2[k](1));
            dSdE12(5) -= s12 * (J_C12(1));
            dSdE12(6) -= s12 * (J_C12(2));
            dSdE12(7) -= s12 * (J_C12(3));
            dSdE12 *= inv_nJ_C12;

            Eigen::Matrix<double, 1, 9> dSdE13;
            dSdE13 << x1[k](0) * x3[k](0), x1[k](0) * x3[k](1), x1[k](0), x1[k](1) * x3[k](0), x1[k](1) * x3[k](1),
                x1[k](1), x3[k](0), x3[k](1), 1.0;
            const double s13 = C13 * inv_nJ_C13 * inv_nJ_C13;
            dSdE13(0) -= s13 * (J_C13(2) * x1[k](0) + J_C13(0) * x3[k](0));
            dSdE13(1) -= s13 * (J_C13(3) * x1[k](0) + J_C13(0) * x3[k](1));
            dSdE13(2) -= s13 * (J_C13(0));
            dSdE13(3) -= s13 * (J_C13(2) * x1[k](1) + J_C13(1) * x3[k](0));
            dSdE13(4) -= s13 * (J_C13(3) * x1[k](1) + J_C13(1) * x3[k](1));
            dSdE13(5) -= s13 * (J_C13(1));
            dSdE13(6) -= s13 * (J_C13(2));
            dSdE13(7) -= s13 * (J_C13(3));
            dSdE13 *= inv_nJ_C13;
            
            Eigen::Matrix<double, 1, 9> dSdE23;
            dSdE23 << x2[k](0) * x3[k](0), x2[k](0) * x3[k](1), x2[k](0), x2[k](1) * x3[k](0), x2[k](1) * x3[k](1),
                x2[k](1), x3[k](0), x3[k](1), 1.0;
            const double s23 = C23 * inv_nJ_C23 * inv_nJ_C23;
            dSdE23(0) -= s23 * (J_C23(2) * x2[k](0) + J_C23(0) * x3[k](0));
            dSdE23(1) -= s23 * (J_C23(3) * x2[k](0) + J_C23(0) * x3[k](1));
            dSdE23(2) -= s23 * (J_C23(0));
            dSdE23(3) -= s23 * (J_C23(2) * x2[k](1) + J_C23(1) * x3[k](0));
            dSdE23(4) -= s23 * (J_C23(3) * x2[k](1) + J_C23(1) * x3[k](1));
            dSdE23(5) -= s23 * (J_C23(1));
            dSdE23(6) -= s23 * (J_C23(2));
            dSdE23(7) -= s23 * (J_C23(3));
            dSdE23 *= inv_nJ_C23;
            
            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 11> J12;
            J12.block<1, 3>(0, 0) = dSdE12 * dE12dr12;
            J12.block<1, 2>(0, 3) = dSdE12 * dE12dt12;
            J12.block<1, 6>(0, 6).setZero();

            Eigen::Matrix<double, 1, 11> J13;
            J13.block<1, 5>(0, 0).setZero();
            J13.block<1, 3>(0, 5) = dSdE13 * dE13dr13;
            J13.block<1, 3>(0, 8) = dSdE13 * dE13dt13;

            Eigen::Matrix<double, 1, 11> J23;
            J23.block<1, 3>(0, 0) = dSdE23 * dE23dr12;
            J23.block<1, 2>(0, 3) = dSdE23 * dE23dt12;
            J23.block<1, 3>(0, 5) = dSdE23 * dE23dr13;
            J23.block<1, 3>(0, 8) = dSdE23 * dE23dt13;


            // Accumulate into Jtr
            Jtr += weight12 * C12 * inv_nJ_C12 * J12.transpose();
            Jtr += weight13 * C13 * inv_nJ_C13 * J13.transpose();
            Jtr += weight23 * C23 * inv_nJ_C23 * J23.transpose();

            for (int row = 0; row < 11; row++)
                for (int col = 0; col < 11; col++)
                    if (row >= col) {
                        JtJ(row, col) += weight12 * (J12(row) * J12(col));
                        JtJ(row, col) += weight13 * (J13(row) * J13(col));
                        JtJ(row, col) += weight23 * (J23(row) * J23(col));
                    }
        }
        return num_residuals;
    }

    ThreeViewCameraPose step(Eigen::Matrix<double, 11, 1> dp, const ThreeViewCameraPose &three_view_pose) const {
        CameraPose pose12_new, pose13_new;

        pose12_new.q = quat_step_post(three_view_pose.pose12.q, dp.block<3, 1>(0, 0));
        pose12_new.t = three_view_pose.pose12.t + tangent_basis * dp.block<2, 1>(3, 0);

        pose13_new.q = quat_step_post(three_view_pose.pose13.q, dp.block<3, 1>(5, 0));
        pose13_new.t = three_view_pose.pose13.t + dp.block<3, 1>(8, 0);

        ThreeViewCameraPose three_view_pose_new(pose12_new, pose13_new);
        return three_view_pose_new;
    }
    typedef ThreeViewCameraPose param_t;
    static constexpr size_t num_params = 11;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class ThreeViewSharedFocalRelativePoseJacobianAccumulator {
  public:
    ThreeViewSharedFocalRelativePoseJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                                        const std::vector<Point2D> &points2D_2,
                                                        const std::vector<Point2D> &points2D_3,
                                                        const LossFunction &l,
                                                        const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), x3(points2D_3), loss_fn(l), weights(w) {}

    double residual(const ImageTriplet &image_triplet) const {
        Eigen::Matrix3d F12, F13, F23;
        Eigen::DiagonalMatrix<double, 3> K_inv(1, 1, image_triplet.camera.focal());
        essential_from_motion(image_triplet.poses.pose12, &F12);
        essential_from_motion(image_triplet.poses.pose13, &F13);
        essential_from_motion(image_triplet.poses.pose23(), &F23);
        F12 = K_inv * F12 * K_inv;
        F13 = K_inv * F13 * K_inv;
        F23 = K_inv * F23 * K_inv;

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            // F12
            double C12 = x2[k].homogeneous().dot(F12 * x1[k].homogeneous());
            double nJc12_sq = (F12.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                              (F12.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();
            double r12_sq = (C12 * C12) / nJc12_sq;
            double loss12 = loss_fn.loss(r12_sq);
            if (loss12 == 0.0)
                continue;

            // F13
            double C13 = x3[k].homogeneous().dot(F13 * x1[k].homogeneous());
            double nJc13_sq = (F13.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                              (F13.block<3, 2>(0, 0).transpose() * x3[k].homogeneous()).squaredNorm();
            
            double r13_sq = (C13 * C13) / nJc13_sq;
            double loss13 = loss_fn.loss(r13_sq);
            if (loss13 == 0.0)
                continue;
            
            // F23
            double C23 = x3[k].homogeneous().dot(F23 * x2[k].homogeneous());
            double nJc23_sq = (F23.block<2, 3>(0, 0) * x2[k].homogeneous()).squaredNorm() +
                              (F23.block<3, 2>(0, 0).transpose() * x3[k].homogeneous()).squaredNorm();

            double r23_sq = (C23 * C23) / nJc23_sq;
            double loss23 = loss_fn.loss(r23_sq);
            if (loss23 == 0.0)
                continue;

            cost += weights[k] * loss12;
            cost += weights[k] * loss13;
            cost += weights[k] * loss23;
        }

        return cost;
    }

//    double residual(const ImageTriplet &image_triplet, size_t k) const {
//        Eigen::Matrix3d F12, F13, F23;
//        Eigen::DiagonalMatrix<double, 3> K_inv(1, 1, image_triplet.camera.focal());
//        essential_from_motion(image_triplet.poses.pose12, &F12);
//        essential_from_motion(image_triplet.poses.pose13, &F13);
//        essential_from_motion(image_triplet.poses.pose23(), &F23);
//        F12 = K_inv * F12 * K_inv;
//        F13 = K_inv * F13 * K_inv;
//        F23 = K_inv * F23 * K_inv;
//
//        double cost = 0.0;
//            // F12
//            double C12 = x2[k].homogeneous().dot(F12 * x1[k].homogeneous());
//            double nJc12_sq = (F12.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
//                              (F12.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();
//            double r12_sq = (C12 * C12) / nJc12_sq;
//            double loss12 = loss_fn.loss(r12_sq);
//            if (loss12 == 0.0)
//                return 0.0;
//
//            // F13
//            double C13 = x3[k].homogeneous().dot(F13 * x1[k].homogeneous());
//            double nJc13_sq = (F13.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
//                              (F13.block<3, 2>(0, 0).transpose() * x3[k].homogeneous()).squaredNorm();
//
//            double r13_sq = (C13 * C13) / nJc13_sq;
//            double loss13 = loss_fn.loss(r13_sq);
//            if (loss13 == 0.0)
//                return 0.0;
//
//            // F23
//            double C23 = x3[k].homogeneous().dot(F23 * x2[k].homogeneous());
//            double nJc23_sq = (F23.block<2, 3>(0, 0) * x2[k].homogeneous()).squaredNorm() +
//                              (F23.block<3, 2>(0, 0).transpose() * x3[k].homogeneous()).squaredNorm();
//
//            double r23_sq = (C23 * C23) / nJc23_sq;
//            double loss23 = loss_fn.loss(r23_sq);
//            if (loss23 == 0.0)
//                return 0.0;
//
//            cost += weights[k] * loss12;
//            cost += weights[k] * loss13;
//            cost += weights[k] * loss23;
//
//        return cost;
//    }

    size_t accumulate(const ImageTriplet &image_triplet, Eigen::Matrix<double, 12, 12> &JtJ,
                      Eigen::Matrix<double, 12, 1> &Jtr) {
        // We use tangent bases for t12 and direct vector for t23
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        
        CameraPose pose12 = image_triplet.poses.pose12;
        
        if (std::abs(pose12.t.x()) < std::abs(pose12.t.y())) {
            // x < y
            if (std::abs(pose12.t.x()) < std::abs(pose12.t.z())) {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(pose12.t.y()) < std::abs(pose12.t.z())) {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(pose12.t).normalized();

        Eigen::Matrix3d E12, R12, E13, R13, E23, R23, F12, F13, F23;

        R12 = pose12.R();
        essential_from_motion(pose12, &E12);
        
        R13 = image_triplet.poses.pose13.R();
        essential_from_motion(image_triplet.poses.pose13, &E13);

        CameraPose pose23 = image_triplet.poses.pose23();
        R23 = pose23.R();
        essential_from_motion(pose23, &E23);

        double focal = image_triplet.camera.focal();

        Eigen::DiagonalMatrix<double, 3> K_inv(1.0, 1.0, focal);
        F12 = K_inv * E12 * K_inv;
        F13 = K_inv * E13 * K_inv;
        F23 = K_inv * E23 * K_inv;

        Eigen::Matrix<double, 9, 1> dF12df, dF13df, dF23df;
        dF12df << 0.0, 0.0, E12(2, 0), 0.0, 0.0, E12(2, 1), E12(0, 2), E12(1, 2), 2 * E12(2, 2) * focal;
        dF13df << 0.0, 0.0, E13(2, 0), 0.0, 0.0, E13(2, 1), E13(0, 2), E13(1, 2), 2 * E13(2, 2) * focal;
        dF23df << 0.0, 0.0, E23(2, 0), 0.0, 0.0, E23(2, 1), E23(0, 2), E23(1, 2), 2 * E23(2, 2) * focal;

        Eigen::Matrix<double, 1, 9> dFdE;
        dFdE << 1.0, 1.0, focal, 1.0, 1.0, focal, focal, focal, focal * focal;

        // Matrices contain the jacobians of E12 w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dE12dr12;
        Eigen::Matrix<double, 9, 2> dE12dt12;

        // Each column is vec(E12*skew(e_k)) where e_k is k:th basis vector
        dE12dr12.block<3, 1>(0, 0).setZero();
        dE12dr12.block<3, 1>(0, 1) = -E12.col(2);
        dE12dr12.block<3, 1>(0, 2) = E12.col(1);
        dE12dr12.block<3, 1>(3, 0) = E12.col(2);
        dE12dr12.block<3, 1>(3, 1).setZero();
        dE12dr12.block<3, 1>(3, 2) = -E12.col(0);
        dE12dr12.block<3, 1>(6, 0) = -E12.col(1);
        dE12dr12.block<3, 1>(6, 1) = E12.col(0);
        dE12dr12.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(tangent_basis[k])*R12)
        dE12dt12.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R12.col(0));
        dE12dt12.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R12.col(0));
        dE12dt12.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R12.col(1));
        dE12dt12.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R12.col(1));
        dE12dt12.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R12.col(2));
        dE12dt12.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R12.col(2));
        
        // Matrices contain the jacobians of E12 w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dE13dr13;
        Eigen::Matrix<double, 9, 3> dE13dt13;
        
        // Each column is vec(E13*skew(e_k)) where e_k is k:th basis vector
        dE13dr13.block<3, 1>(0, 0).setZero();
        dE13dr13.block<3, 1>(0, 1) = -E13.col(2);
        dE13dr13.block<3, 1>(0, 2) = E13.col(1);
        dE13dr13.block<3, 1>(3, 0) = E13.col(2);
        dE13dr13.block<3, 1>(3, 1).setZero();
        dE13dr13.block<3, 1>(3, 2) = -E13.col(0);
        dE13dr13.block<3, 1>(6, 0) = -E13.col(1);
        dE13dr13.block<3, 1>(6, 1) = E13.col(0);
        dE13dr13.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(e_k)*R13) where e_k is k:th basis vector
        Eigen::Matrix3d dE13dt13_0, dE13dt13_1, dE13dt13_2;
        dE13dt13_0.row(0).setZero();
        dE13dt13_0.row(1) = -R13.row(2);
        dE13dt13_0.row(2) = R13.row(1);

        dE13dt13_1.row(0) = R13.row(2);
        dE13dt13_1.row(1).setZero();
        dE13dt13_1.row(2) = - R13.row(0);

        dE13dt13_2.row(0) = - R13.row(1);
        dE13dt13_2.row(1) = R13.row(0);
        dE13dt13_2.row(2).setZero();

        dE13dt13.col(0) = Eigen::Map<Eigen::VectorXd>(dE13dt13_0.data(), dE13dt13_0.size());
        dE13dt13.col(1) = Eigen::Map<Eigen::VectorXd>(dE13dt13_1.data(), dE13dt13_1.size());
        dE13dt13.col(2) = Eigen::Map<Eigen::VectorXd>(dE13dt13_2.data(), dE13dt13_2.size());

        //TODO: this part calculates dE23dX and is not optimized yet

        // define skew(e_k)
        Eigen::Matrix3d b_0 = skew(Eigen::Vector3d::UnitX());
        Eigen::Matrix3d b_1 = skew(Eigen::Vector3d::UnitY());
        Eigen::Matrix3d b_2 = skew(Eigen::Vector3d::UnitZ());

        Eigen::Matrix3d dE23dr12_0, dE23dr12_1, dE23dr12_2;
        dE23dr12_0 = skew(R13 * b_0 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_0 * R12.transpose();
        dE23dr12_1 = skew(R13 * b_1 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_1 * R12.transpose();
        dE23dr12_2 = skew(R13 * b_2 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_2 * R12.transpose();

        Eigen::Matrix3d dE23dr13_0, dE23dr13_1, dE23dr13_2;
        dE23dr13_0 = - dE23dr12_0;
        dE23dr13_1 = - dE23dr12_1;
        dE23dr13_2 = - dE23dr12_2;

        // dE23dt12 = skew(tangent_basis_k) * R23
        Eigen::Matrix3d dE23dt12_0, dE23dt12_1;
        dE23dt12_0 = - skew(R23 * tangent_basis.col(0)) * R23;
        dE23dt12_1 = - skew(R23 * tangent_basis.col(1)) * R23;

        // dE23dt13 = skew(e_k) * R23
        Eigen::Matrix3d dE23dt13_0, dE23dt13_1, dE23dt13_2;
        dE23dt13_0.row(0).setZero();
        dE23dt13_0.row(1) = -R23.row(2);
        dE23dt13_0.row(2) = R23.row(1);

        dE23dt13_1.row(0) = R23.row(2);
        dE23dt13_1.row(1).setZero();
        dE23dt13_1.row(2) = - R23.row(0);

        dE23dt13_2.row(0) = - R23.row(1);
        dE23dt13_2.row(1) = R23.row(0);
        dE23dt13_2.row(2).setZero();

        Eigen::Matrix<double, 9, 3> dE23dr12;
        Eigen::Matrix<double, 9, 3> dE23dr13;
        Eigen::Matrix<double, 9, 2> dE23dt12;
        Eigen::Matrix<double, 9, 3> dE23dt13;

        dE23dr12.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dr12_0.data(), dE23dr12_0.size());
        dE23dr12.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dr12_1.data(), dE23dr12_1.size());
        dE23dr12.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dr12_2.data(), dE23dr12_2.size());

        dE23dr13.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dr13_0.data(), dE23dr13_0.size());
        dE23dr13.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dr13_1.data(), dE23dr13_1.size());
        dE23dr13.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dr13_2.data(), dE23dr13_2.size());

        dE23dt12.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dt12_0.data(), dE23dt12_0.size());
        dE23dt12.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dt12_1.data(), dE23dt12_1.size());

        dE23dt13.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dt13_0.data(), dE23dt13_0.size());
        dE23dt13.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dt13_1.data(), dE23dt13_1.size());
        dE23dt13.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dt13_2.data(), dE23dt13_2.size());


        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C12 = x2[k].homogeneous().dot(F12 * x1[k].homogeneous());
            double C13 = x3[k].homogeneous().dot(F13 * x1[k].homogeneous());
            double C23 = x3[k].homogeneous().dot(F23 * x2[k].homogeneous());

            // J_C12 is the Jacobian of the epipolar constraint w.r12.t. the image points
            Eigen::Vector4d J_C12;
            J_C12 << F12.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), F12.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C12 = J_C12.norm();
            const double inv_nJ_C12 = 1.0 / nJ_C12;
            const double r12 = C12 * inv_nJ_C12;
            const double weight12 = weights[k] * loss_fn.weight(r12 * r12);
            if (weight12 == 0.0) {
                continue;
            }
                        
            Eigen::Vector4d J_C13;
            J_C13 << F13.block<3, 2>(0, 0).transpose() * x3[k].homogeneous(), F13.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C13 = J_C13.norm();
            const double inv_nJ_C13 = 1.0 / nJ_C13;
            const double r13 = C13 * inv_nJ_C13;

            const double weight13 = weights[k] * loss_fn.weight(r13 * r13);
            if (weight13 == 0.0) {
                continue;
            }
            
            Eigen::Vector4d J_C23;
            J_C23 << F23.block<3, 2>(0, 0).transpose() * x3[k].homogeneous(), F23.block<2, 3>(0, 0) * x2[k].homogeneous();
            const double nJ_C23 = J_C23.norm();
            const double inv_nJ_C23 = 1.0 / nJ_C23;
            const double r23 = C23 * inv_nJ_C23;

            const double weight23 = weights[k] * loss_fn.weight(r23 * r23);
            if (weight23 == 0.0) {
                continue;
            }
                        
            num_residuals += 3;

            // Compute Jacobian of Sampson error w.r12.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dSdF12;
            dSdF12 << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s12 = C12 * inv_nJ_C12 * inv_nJ_C12;
            dSdF12(0) -= s12 * (J_C12(2) * x1[k](0) + J_C12(0) * x2[k](0));
            dSdF12(1) -= s12 * (J_C12(3) * x1[k](0) + J_C12(0) * x2[k](1));
            dSdF12(2) -= s12 * (J_C12(0));
            dSdF12(3) -= s12 * (J_C12(2) * x1[k](1) + J_C12(1) * x2[k](0));
            dSdF12(4) -= s12 * (J_C12(3) * x1[k](1) + J_C12(1) * x2[k](1));
            dSdF12(5) -= s12 * (J_C12(1));
            dSdF12(6) -= s12 * (J_C12(2));
            dSdF12(7) -= s12 * (J_C12(3));
            dSdF12 *= inv_nJ_C12;

            Eigen::Matrix<double, 1, 9> dSdF13;
            dSdF13 << x1[k](0) * x3[k](0), x1[k](0) * x3[k](1), x1[k](0), x1[k](1) * x3[k](0), x1[k](1) * x3[k](1),
                x1[k](1), x3[k](0), x3[k](1), 1.0;
            const double s13 = C13 * inv_nJ_C13 * inv_nJ_C13;
            dSdF13(0) -= s13 * (J_C13(2) * x1[k](0) + J_C13(0) * x3[k](0));
            dSdF13(1) -= s13 * (J_C13(3) * x1[k](0) + J_C13(0) * x3[k](1));
            dSdF13(2) -= s13 * (J_C13(0));
            dSdF13(3) -= s13 * (J_C13(2) * x1[k](1) + J_C13(1) * x3[k](0));
            dSdF13(4) -= s13 * (J_C13(3) * x1[k](1) + J_C13(1) * x3[k](1));
            dSdF13(5) -= s13 * (J_C13(1));
            dSdF13(6) -= s13 * (J_C13(2));
            dSdF13(7) -= s13 * (J_C13(3));
            dSdF13 *= inv_nJ_C13;
            
            Eigen::Matrix<double, 1, 9> dSdF23;
            dSdF23 << x2[k](0) * x3[k](0), x2[k](0) * x3[k](1), x2[k](0), x2[k](1) * x3[k](0), x2[k](1) * x3[k](1),
                x2[k](1), x3[k](0), x3[k](1), 1.0;
            const double s23 = C23 * inv_nJ_C23 * inv_nJ_C23;
            dSdF23(0) -= s23 * (J_C23(2) * x2[k](0) + J_C23(0) * x3[k](0));
            dSdF23(1) -= s23 * (J_C23(3) * x2[k](0) + J_C23(0) * x3[k](1));
            dSdF23(2) -= s23 * (J_C23(0));
            dSdF23(3) -= s23 * (J_C23(2) * x2[k](1) + J_C23(1) * x3[k](0));
            dSdF23(4) -= s23 * (J_C23(3) * x2[k](1) + J_C23(1) * x3[k](1));
            dSdF23(5) -= s23 * (J_C23(1));
            dSdF23(6) -= s23 * (J_C23(2));
            dSdF23(7) -= s23 * (J_C23(3));
            dSdF23 *= inv_nJ_C23;
            
            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 12> J12;
            J12.block<1, 3>(0, 0) = dSdF12.cwiseProduct(dFdE) * dE12dr12;
            J12.block<1, 2>(0, 3) = dSdF12.cwiseProduct(dFdE) * dE12dt12;
            J12.block<1, 6>(0, 6).setZero();
            J12(0, 11) = dSdF12 * dF12df;

            Eigen::Matrix<double, 1, 12> J13;
            J13.block<1, 5>(0, 0).setZero();
            J13.block<1, 3>(0, 5) = dSdF13.cwiseProduct(dFdE) * dE13dr13;
            J13.block<1, 3>(0, 8) = dSdF13.cwiseProduct(dFdE) * dE13dt13;
            J13(0, 11) = dSdF13 * dF13df;

            Eigen::Matrix<double, 1, 12> J23;
            J23.block<1, 3>(0, 0) = dSdF23.cwiseProduct(dFdE) * dE23dr12;
            J23.block<1, 2>(0, 3) = dSdF23.cwiseProduct(dFdE) * dE23dt12;
            J23.block<1, 3>(0, 5) = dSdF23.cwiseProduct(dFdE) * dE23dr13;
            J23.block<1, 3>(0, 8) = dSdF23.cwiseProduct(dFdE) * dE23dt13;
            J23(0, 11) = dSdF23 * dF23df;

//            Eigen::Matrix<double, 1, 12> num_J;
//            Eigen::Matrix<double, 12, 1> dp;
//            double eps = 1.0e-8;
//            for (int j = 0; j < 12; ++j){
//                dp.setZero();
//                dp(j, 0) = eps;
//                ImageTriplet step_image_triplet = step(dp, image_triplet);
//                dp.setZero();
//                num_J(0, j) = (residual(step_image_triplet, k) - residual(image_triplet, k)) / eps;
//            }
//
//            std::cout << "Sym J: " << 2 * (weight12 * C12 * inv_nJ_C12 * J12 + weight13 * C13 * inv_nJ_C13 * J13 + weight23 * C23 * inv_nJ_C23 * J23) << std::endl;
//            std::cout << "Num J: " << num_J << std::endl;

            // Accumulate into Jtr
            Jtr += weight12 * C12 * inv_nJ_C12 * J12.transpose();
            Jtr += weight13 * C13 * inv_nJ_C13 * J13.transpose();
            Jtr += weight23 * C23 * inv_nJ_C23 * J23.transpose();

            for (int row = 0; row < 12; row++)
                for (int col = 0; col < 12; col++)
                    if (row >= col) {
                        JtJ(row, col) += weight12 * (J12(row) * J12(col));
                        JtJ(row, col) += weight13 * (J13(row) * J13(col));
                        JtJ(row, col) += weight23 * (J23(row) * J23(col));
                    }
        }
        return num_residuals;
    }

    ImageTriplet step(Eigen::Matrix<double, 12, 1> dp, const ImageTriplet &image_triplet) const {
        CameraPose pose12_new, pose13_new;

        pose12_new.q = quat_step_post(image_triplet.poses.pose12.q, dp.block<3, 1>(0, 0));
        pose12_new.t = image_triplet.poses.pose12.t + tangent_basis * dp.block<2, 1>(3, 0);

        pose13_new.q = quat_step_post(image_triplet.poses.pose13.q, dp.block<3, 1>(5, 0));
        pose13_new.t = image_triplet.poses.pose13.t + dp.block<3, 1>(8, 0);

        Camera camera_new = Camera("SIMPLE_PINHOLE", std::vector<double>{image_triplet.camera.focal()+ dp(11,0), 0.0, 0.0}, -1, -1);

        ImageTriplet image_triplet_new(ThreeViewCameraPose(pose12_new, pose13_new), camera_new);
        return image_triplet_new;
    }
    typedef ImageTriplet param_t;
    static constexpr size_t num_params = 12;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class SharedFocalRelativePoseJacobianAccumulator {
  public:
    SharedFocalRelativePoseJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                               const std::vector<Point2D> &points2D_2, const LossFunction &l,
                                               const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const ImagePair &image_pair) const {
        Eigen::Matrix3d E;
        essential_from_motion(image_pair.pose, &E);
        Eigen::Matrix3d K_inv;
        K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, image_pair.camera1.focal();

        Eigen::Matrix3d F = K_inv * (E * K_inv);

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const ImagePair &image_pair, Eigen::Matrix<double, 6, 6> &JtJ, Eigen::Matrix<double, 6, 1> &Jtr) {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(image_pair.pose.t.x()) < std::abs(image_pair.pose.t.y())) {
            // x < y
            if (std::abs(image_pair.pose.t.x()) < std::abs(image_pair.pose.t.z())) {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(image_pair.pose.t.y()) < std::abs(image_pair.pose.t.z())) {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(image_pair.pose.t).normalized();

        double focal = image_pair.camera1.focal();
        Eigen::Matrix3d K_inv;
        K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, focal;

        Eigen::Matrix3d E, R;
        R = image_pair.pose.R();
        essential_from_motion(image_pair.pose, &E);
        Eigen::Matrix3d F = K_inv * (E * K_inv);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        dR.row(2) *= focal;
        dR.row(5) *= focal;
        dR.row(6) *= focal;
        dR.row(7) *= focal;
        dR.row(8) *= focal * focal;

        // Each column is vec(skew(tangent_basis[k])*R)
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        dt.row(2) *= focal;
        dt.row(5) *= focal;
        dt.row(6) *= focal;
        dt.row(7) *= focal;
        dt.row(8) *= focal * focal;

        Eigen::Matrix<double, 9, 1> df;

        df << 0.0, 0.0, E(2, 0), 0.0, 0.0, E(2, 1), E(0, 2), E(1, 2), 2 * E(2, 2) * focal;

        size_t num_residuals = 0;

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), F.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r * r);
            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 6> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;
            J(0, 5) = dF * df;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            JtJ(0, 0) += weight * (J(0) * J(0));
            JtJ(1, 0) += weight * (J(1) * J(0));
            JtJ(1, 1) += weight * (J(1) * J(1));
            JtJ(2, 0) += weight * (J(2) * J(0));
            JtJ(2, 1) += weight * (J(2) * J(1));
            JtJ(2, 2) += weight * (J(2) * J(2));
            JtJ(3, 0) += weight * (J(3) * J(0));
            JtJ(3, 1) += weight * (J(3) * J(1));
            JtJ(3, 2) += weight * (J(3) * J(2));
            JtJ(3, 3) += weight * (J(3) * J(3));
            JtJ(4, 0) += weight * (J(4) * J(0));
            JtJ(4, 1) += weight * (J(4) * J(1));
            JtJ(4, 2) += weight * (J(4) * J(2));
            JtJ(4, 3) += weight * (J(4) * J(3));
            JtJ(4, 4) += weight * (J(4) * J(4));
            JtJ(5, 0) += weight * (J(5) * J(0));
            JtJ(5, 1) += weight * (J(5) * J(1));
            JtJ(5, 2) += weight * (J(5) * J(2));
            JtJ(5, 3) += weight * (J(5) * J(3));
            JtJ(5, 4) += weight * (J(5) * J(4));
            JtJ(5, 5) += weight * (J(5) * J(5));
        }
        return num_residuals;
    }

    ImagePair step(Eigen::Matrix<double, 6, 1> dp, const ImagePair &image_pair) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(image_pair.pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = image_pair.pose.t + tangent_basis * dp.block<2, 1>(3, 0);

        Camera camera_new =
            Camera("SIMPLE_PINHOLE",
                   std::vector<double>{std::max(image_pair.camera1.focal() + dp(5, 0), 0.0), 0.0, 0.0}, -1, -1);
        ImagePair calib_pose_new(pose_new, camera_new, camera_new);
        return calib_pose_new;
    }
    typedef ImagePair param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

template <typename LossFunction, typename ResidualWeightVectors = UniformWeightVectors>
class GeneralizedRelativePoseJacobianAccumulator {
  public:
    GeneralizedRelativePoseJacobianAccumulator(const std::vector<PairwiseMatches> &pairwise_matches,
                                               const std::vector<CameraPose> &camera1_ext,
                                               const std::vector<CameraPose> &camera2_ext, const LossFunction &l,
                                               const ResidualWeightVectors &w = ResidualWeightVectors())
        : matches(pairwise_matches), rig1_poses(camera1_ext), rig2_poses(camera2_ext), loss_fn(l), weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0.0;
        for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
            const PairwiseMatches &m = matches[match_k];
            Eigen::Vector4d q1 = rig1_poses[m.cam_id1].q;
            Eigen::Vector3d t1 = rig1_poses[m.cam_id1].t;

            Eigen::Vector4d q2 = rig2_poses[m.cam_id2].q;
            Eigen::Vector3d t2 = rig2_poses[m.cam_id2].t;

            CameraPose relpose;
            relpose.q = quat_multiply(q2, quat_multiply(pose.q, quat_conj(q1)));
            relpose.t = t2 + quat_rotate(q2, pose.t) - relpose.rotate(t1);
            RelativePoseJacobianAccumulator<LossFunction, typename ResidualWeightVectors::value_type> accum(
                m.x1, m.x2, loss_fn, weights[match_k]);
            cost += accum.residual(relpose);
        }
        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        Eigen::Matrix3d R = pose.R();
        size_t num_residuals = 0;
        for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
            const PairwiseMatches &m = matches[match_k];

            // Cameras are
            // [R1 t1]
            // [R2 t2] * [R t; 0 1] = [R2*R t2+R2*t]

            // Relative pose is
            // [R2*R*R1' t2+R2*t-R2*R*R1'*t1]
            // Essential matrix is
            // [t2]_x*R2*R*R1' + [R2*t]_x*R2*R*R1' - R2*R*R1'*[t1]_x

            Eigen::Vector4d q1 = rig1_poses[m.cam_id1].q;
            Eigen::Matrix3d R1 = quat_to_rotmat(q1);
            Eigen::Vector3d t1 = rig1_poses[m.cam_id1].t;

            Eigen::Vector4d q2 = rig2_poses[m.cam_id2].q;
            Eigen::Matrix3d R2 = quat_to_rotmat(q2);
            Eigen::Vector3d t2 = rig2_poses[m.cam_id2].t;

            CameraPose relpose;
            relpose.q = quat_multiply(q2, quat_multiply(pose.q, quat_conj(q1)));
            relpose.t = t2 + R2 * pose.t - relpose.rotate(t1);
            Eigen::Matrix3d E;
            essential_from_motion(relpose, &E);

            Eigen::Matrix3d R2R = R2 * R;
            Eigen::Vector3d Rt = R.transpose() * pose.t;

            // The messy expressions below compute
            // dRdw = [vec(S1) vec(S2) vec(S3)];
            // dR = (kron(R1,skew(t2)*R2R+ R2*skew(t)*R) + kron(skew(t1)*R1,R2*R)) * dRdw
            // dt = [vec(R2*R*S1*R1.') vec(R2*R*S2*R1.') vec(R2*R*S3*R1.')]

            // TODO: Replace with something nice
            Eigen::Matrix<double, 9, 3> dR;
            Eigen::Matrix<double, 9, 3> dt;
            dR(0, 0) = R2R(0, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R2R(0, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
                       R1(0, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(0, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
            dR(0, 1) = R2R(0, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R2R(0, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R1(0, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(0, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(0, 2) = R2R(0, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) -
                       R2R(0, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R1(0, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
                       R1(0, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(1, 0) = R2R(1, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R2R(1, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
                       R1(0, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(0, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
            dR(1, 1) = R2R(1, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R2R(1, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R1(0, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(0, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(1, 2) = R2R(1, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) -
                       R2R(1, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R1(0, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
                       R1(0, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(2, 0) = R2R(2, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R2R(2, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
                       R1(0, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(0, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
            dR(2, 1) = R2R(2, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R2R(2, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R1(0, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(0, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(2, 2) = R2R(2, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) -
                       R2R(2, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R1(0, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
                       R1(0, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(3, 0) = R2R(0, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R2R(0, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
                       R1(1, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(1, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
            dR(3, 1) = R2R(0, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) -
                       R2R(0, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R1(1, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(1, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(3, 2) = R2R(0, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R2R(0, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R1(1, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
                       R1(1, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(4, 0) = R2R(1, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R2R(1, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
                       R1(1, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(1, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
            dR(4, 1) = R2R(1, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) -
                       R2R(1, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R1(1, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(1, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(4, 2) = R2R(1, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R2R(1, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R1(1, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
                       R1(1, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(5, 0) = R2R(2, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R2R(2, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
                       R1(1, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(1, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
            dR(5, 1) = R2R(2, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) -
                       R2R(2, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R1(1, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(1, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(5, 2) = R2R(2, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R2R(2, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R1(1, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
                       R1(1, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(6, 0) = R2R(0, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R2R(0, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
                       R1(2, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(2, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
            dR(6, 1) = R2R(0, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R2R(0, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R1(2, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(2, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(6, 2) = R2R(0, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) -
                       R2R(0, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R1(2, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
                       R1(2, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(7, 0) = R2R(1, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R2R(1, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
                       R1(2, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(2, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
            dR(7, 1) = R2R(1, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R2R(1, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R1(2, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(2, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(7, 2) = R2R(1, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) -
                       R2R(1, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R1(2, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
                       R1(2, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(8, 0) = R2R(2, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R2R(2, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
                       R1(2, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(2, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
            dR(8, 1) = R2R(2, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R2R(2, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R1(2, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(2, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(8, 2) = R2R(2, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) -
                       R2R(2, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R1(2, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
                       R1(2, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dt(0, 0) = R2R(0, 2) * R1(0, 1) - R2R(0, 1) * R1(0, 2);
            dt(0, 1) = R2R(0, 0) * R1(0, 2) - R2R(0, 2) * R1(0, 0);
            dt(0, 2) = R2R(0, 1) * R1(0, 0) - R2R(0, 0) * R1(0, 1);
            dt(1, 0) = R2R(1, 2) * R1(0, 1) - R2R(1, 1) * R1(0, 2);
            dt(1, 1) = R2R(1, 0) * R1(0, 2) - R2R(1, 2) * R1(0, 0);
            dt(1, 2) = R2R(1, 1) * R1(0, 0) - R2R(1, 0) * R1(0, 1);
            dt(2, 0) = R2R(2, 2) * R1(0, 1) - R2R(2, 1) * R1(0, 2);
            dt(2, 1) = R2R(2, 0) * R1(0, 2) - R2R(2, 2) * R1(0, 0);
            dt(2, 2) = R2R(2, 1) * R1(0, 0) - R2R(2, 0) * R1(0, 1);
            dt(3, 0) = R2R(0, 2) * R1(1, 1) - R2R(0, 1) * R1(1, 2);
            dt(3, 1) = R2R(0, 0) * R1(1, 2) - R2R(0, 2) * R1(1, 0);
            dt(3, 2) = R2R(0, 1) * R1(1, 0) - R2R(0, 0) * R1(1, 1);
            dt(4, 0) = R2R(1, 2) * R1(1, 1) - R2R(1, 1) * R1(1, 2);
            dt(4, 1) = R2R(1, 0) * R1(1, 2) - R2R(1, 2) * R1(1, 0);
            dt(4, 2) = R2R(1, 1) * R1(1, 0) - R2R(1, 0) * R1(1, 1);
            dt(5, 0) = R2R(2, 2) * R1(1, 1) - R2R(2, 1) * R1(1, 2);
            dt(5, 1) = R2R(2, 0) * R1(1, 2) - R2R(2, 2) * R1(1, 0);
            dt(5, 2) = R2R(2, 1) * R1(1, 0) - R2R(2, 0) * R1(1, 1);
            dt(6, 0) = R2R(0, 2) * R1(2, 1) - R2R(0, 1) * R1(2, 2);
            dt(6, 1) = R2R(0, 0) * R1(2, 2) - R2R(0, 2) * R1(2, 0);
            dt(6, 2) = R2R(0, 1) * R1(2, 0) - R2R(0, 0) * R1(2, 1);
            dt(7, 0) = R2R(1, 2) * R1(2, 1) - R2R(1, 1) * R1(2, 2);
            dt(7, 1) = R2R(1, 0) * R1(2, 2) - R2R(1, 2) * R1(2, 0);
            dt(7, 2) = R2R(1, 1) * R1(2, 0) - R2R(1, 0) * R1(2, 1);
            dt(8, 0) = R2R(2, 2) * R1(2, 1) - R2R(2, 1) * R1(2, 2);
            dt(8, 1) = R2R(2, 0) * R1(2, 2) - R2R(2, 2) * R1(2, 0);
            dt(8, 2) = R2R(2, 1) * R1(2, 0) - R2R(2, 0) * R1(2, 1);

            for (size_t k = 0; k < m.x1.size(); ++k) {
                double C = m.x2[k].homogeneous().dot(E * m.x1[k].homogeneous());

                // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
                Eigen::Vector4d J_C;
                J_C << E.block<3, 2>(0, 0).transpose() * m.x2[k].homogeneous(),
                    E.block<2, 3>(0, 0) * m.x1[k].homogeneous();
                const double nJ_C = J_C.norm();
                const double inv_nJ_C = 1.0 / nJ_C;
                const double r = C * inv_nJ_C;

                // Compute weight from robust loss function (used in the IRLS)
                const double weight = weights[match_k][k] * loss_fn.weight(r * r);
                if (weight == 0.0) {
                    continue;
                }
                num_residuals++;

                // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
                Eigen::Matrix<double, 1, 9> dF;
                dF << m.x1[k](0) * m.x2[k](0), m.x1[k](0) * m.x2[k](1), m.x1[k](0), m.x1[k](1) * m.x2[k](0),
                    m.x1[k](1) * m.x2[k](1), m.x1[k](1), m.x2[k](0), m.x2[k](1), 1.0;
                const double s = C * inv_nJ_C * inv_nJ_C;
                dF(0) -= s * (J_C(2) * m.x1[k](0) + J_C(0) * m.x2[k](0));
                dF(1) -= s * (J_C(3) * m.x1[k](0) + J_C(0) * m.x2[k](1));
                dF(2) -= s * (J_C(0));
                dF(3) -= s * (J_C(2) * m.x1[k](1) + J_C(1) * m.x2[k](0));
                dF(4) -= s * (J_C(3) * m.x1[k](1) + J_C(1) * m.x2[k](1));
                dF(5) -= s * (J_C(1));
                dF(6) -= s * (J_C(2));
                dF(7) -= s * (J_C(3));
                dF *= inv_nJ_C;

                // and then w.r.t. the pose parameters
                Eigen::Matrix<double, 1, 6> J;
                J.block<1, 3>(0, 0) = dF * dR;
                J.block<1, 3>(0, 3) = dF * dt;

                // Accumulate into JtJ and Jtr
                Jtr += weight * C * inv_nJ_C * J.transpose();
                for (size_t i = 0; i < 6; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        JtJ(i, j) += weight * (J(i) * J(j));
                    }
                }
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &rig1_poses;
    const std::vector<CameraPose> &rig2_poses;
    const LossFunction &loss_fn;
    const ResidualWeightVectors &weights;
};

template <typename LossFunction, typename AbsResidualsVector = UniformWeightVector,
          typename RelResidualsVectors = UniformWeightVectors>
class HybridPoseJacobianAccumulator {
  public:
    HybridPoseJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                  const std::vector<PairwiseMatches> &pairwise_matches,
                                  const std::vector<CameraPose> &map_ext, const LossFunction &l,
                                  const LossFunction &l_epi,
                                  const AbsResidualsVector &weights_abs = AbsResidualsVector(),
                                  const RelResidualsVectors &weights_rel = RelResidualsVectors())
        : abs_pose_accum(points2D, points3D, trivial_camera, l, weights_abs),
          gen_rel_accum(pairwise_matches, map_ext, trivial_rig, l_epi, weights_rel) {
        trivial_camera.model_id = NullCameraModel::model_id;
        trivial_rig.emplace_back();
    }

    double residual(const CameraPose &pose) const {
        return abs_pose_accum.residual(pose) + gen_rel_accum.residual(pose);
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        return abs_pose_accum.accumulate(pose, JtJ, Jtr) + gen_rel_accum.accumulate(pose, JtJ, Jtr);
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    Camera trivial_camera;
    std::vector<CameraPose> trivial_rig;
    CameraJacobianAccumulator<NullCameraModel, LossFunction, AbsResidualsVector> abs_pose_accum;
    GeneralizedRelativePoseJacobianAccumulator<LossFunction, RelResidualsVectors> gen_rel_accum;
};

// This is the SVD factorization proposed by Bartoli and Sturm in
// Non-Linear Estimation of the Fundamental Matrix With Minimal Parameters, PAMI 2004
// Though we do different updates (lie vs the euler angles used in the original paper)
struct FactorizedFundamentalMatrix {
    FactorizedFundamentalMatrix() {}
    FactorizedFundamentalMatrix(const Eigen::Matrix3d &F) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullV | Eigen::ComputeFullU);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        if (U.determinant() < 0) {
            U = -U;
        }
        if (V.determinant() < 0) {
            V = -V;
        }
        qU = rotmat_to_quat(U);
        qV = rotmat_to_quat(V);
        Eigen::Vector3d s = svd.singularValues();
        sigma = s(1) / s(0);
    }
    Eigen::Matrix3d F() const {
        Eigen::Matrix3d U = quat_to_rotmat(qU);
        Eigen::Matrix3d V = quat_to_rotmat(qV);
        return U.col(0) * V.col(0).transpose() + sigma * U.col(1) * V.col(1).transpose();
    }

    Eigen::Vector4d qU, qV;
    double sigma;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class FundamentalJacobianAccumulator {
  public:
    FundamentalJacobianAccumulator(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                   const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const FactorizedFundamentalMatrix &FF) const {
        Eigen::Matrix3d F = FF.F();

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const FactorizedFundamentalMatrix &FF, Eigen::Matrix<double, 7, 7> &JtJ,
                      Eigen::Matrix<double, 7, 1> &Jtr) const {

        const Eigen::Matrix3d F = FF.F();

        // Matrices contain the jacobians of F w.r.t. the factorized fundamental matrix (U,V,sigma)
        const Eigen::Matrix3d U = quat_to_rotmat(FF.qU);
        const Eigen::Matrix3d V = quat_to_rotmat(FF.qV);

        const Eigen::Matrix3d d_sigma = U.col(1) * V.col(1).transpose();
        Eigen::Matrix<double, 9, 7> dF_dparams;
        dF_dparams << 0, F(2, 0), -F(1, 0), 0, F(0, 2), -F(0, 1), d_sigma(0, 0), -F(2, 0), 0, F(0, 0), 0, F(1, 2),
            -F(1, 1), d_sigma(1, 0), F(1, 0), -F(0, 0), 0, 0, F(2, 2), -F(2, 1), d_sigma(2, 0), 0, F(2, 1), -F(1, 1),
            -F(0, 2), 0, F(0, 0), d_sigma(0, 1), -F(2, 1), 0, F(0, 1), -F(1, 2), 0, F(1, 0), d_sigma(1, 1), F(1, 1),
            -F(0, 1), 0, -F(2, 2), 0, F(2, 0), d_sigma(2, 1), 0, F(2, 2), -F(1, 2), F(0, 1), -F(0, 0), 0, d_sigma(0, 2),
            -F(2, 2), 0, F(0, 2), F(1, 1), -F(1, 0), 0, d_sigma(1, 2), F(1, 2), -F(0, 2), 0, F(2, 1), -F(2, 0), 0,
            d_sigma(2, 2);

        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            const double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), F.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r * r);
            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 7> J = dF * dF_dparams;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            for (size_t i = 0; i < 7; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J(i) * J(j));
                }
            }
        }
        return num_residuals;
    }

    FactorizedFundamentalMatrix step(Eigen::Matrix<double, 7, 1> dp, const FactorizedFundamentalMatrix &F) const {
        FactorizedFundamentalMatrix F_new;
        F_new.qU = quat_step_pre(F.qU, dp.block<3, 1>(0, 0));
        F_new.qV = quat_step_pre(F.qV, dp.block<3, 1>(3, 0));
        F_new.sigma = F.sigma + dp(6);
        return F_new;
    }
    typedef FactorizedFundamentalMatrix param_t;
    static constexpr size_t num_params = 7;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

// Non-linear refinement of transfer error |x2 - pi(H*x1)|^2, parameterized by fixing H(2,2) = 1
// I did some preliminary experiments comparing different error functions (e.g. symmetric and transfer)
// as well as other parameterizations (different affine patches, SVD as in Bartoli/Sturm, etc)
// but it does not seem to have a big impact (and is sometimes even worse)
// Implementations of these can be found at https://github.com/vlarsson/homopt
template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class HomographyJacobianAccumulator {
  public:
    HomographyJacobianAccumulator(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                  const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const Eigen::Matrix3d &H) const {
        double cost = 0.0;

        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        for (size_t k = 0; k < x1.size(); ++k) {
            const double x1_0 = x1[k](0), x1_1 = x1[k](1);
            const double x2_0 = x2[k](0), x2_1 = x2[k](1);

            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

            const double r0 = Hx1_0 * inv_Hx1_2 - x2_0;
            const double r1 = Hx1_1 * inv_Hx1_2 - x2_1;
            const double r2 = r0 * r0 + r1 * r1;
            cost += weights[k] * loss_fn.loss(r2);
        }
        return cost;
    }

    size_t accumulate(const Eigen::Matrix3d &H, Eigen::Matrix<double, 8, 8> &JtJ, Eigen::Matrix<double, 8, 1> &Jtr) {
        Eigen::Matrix<double, 2, 8> dH;
        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            const double x1_0 = x1[k](0), x1_1 = x1[k](1);
            const double x2_0 = x2[k](0), x2_1 = x2[k](1);

            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

            const double z0 = Hx1_0 * inv_Hx1_2;
            const double z1 = Hx1_1 * inv_Hx1_2;

            const double r0 = z0 - x2_0;
            const double r1 = z1 - x2_1;
            const double r2 = r0 * r0 + r1 * r1;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r2);
            if (weight == 0.0)
                continue;
            num_residuals++;

            dH << x1_0, 0.0, -x1_0 * z0, x1_1, 0.0, -x1_1 * z0, 1.0, 0.0, // -z0,
                0.0, x1_0, -x1_0 * z1, 0.0, x1_1, -x1_1 * z1, 0.0, 1.0;   // -z1,
            dH = dH * inv_Hx1_2;

            // accumulate into JtJ and Jtr
            Jtr += dH.transpose() * (weight * Eigen::Vector2d(r0, r1));
            for (size_t i = 0; i < 8; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * dH.col(i).dot(dH.col(j));
                }
            }
        }
        return num_residuals;
    }

    Eigen::Matrix3d step(Eigen::Matrix<double, 8, 1> dp, const Eigen::Matrix3d &H) const {
        Eigen::Matrix3d H_new = H;
        Eigen::Map<Eigen::Matrix<double, 8, 1>>(H_new.data()) += dp;
        return H_new;
    }
    typedef Eigen::Matrix3d param_t;
    static constexpr size_t num_params = 8;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class Radial1DJacobianAccumulator {
  public:
    Radial1DJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D), X(points3D), loss_fn(l), weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0.0;
        Eigen::Matrix3d R = pose.R();
        for (size_t k = 0; k < x.size(); ++k) {
            Eigen::Vector2d z = (R * X[k] + pose.t).template topRows<2>().normalized();
            double alpha = z.dot(x[k]);
            // This assumes points will not cross the half-space during optimization
            if (alpha < 0)
                continue;
            double r2 = (alpha * z - x[k]).squaredNorm();
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 5, 5> &JtJ,
                      Eigen::Matrix<double, 5, 1> &Jtr) const {
        Eigen::Matrix3d R = pose.R();
        size_t num_residuals = 0;
        for (size_t k = 0; k < x.size(); ++k) {
            Eigen::Vector3d RX = R * X[k];
            const Eigen::Vector2d z = (RX + pose.t).topRows<2>();

            const double n_z = z.norm();
            const Eigen::Vector2d zh = z / n_z;
            const double alpha = zh.dot(x[k]);
            // This assumes points will not cross the half-space during optimization
            if (alpha < 0)
                continue;

            // Setup residual
            Eigen::Vector2d r = alpha * zh - x[k];
            const double r_squared = r.squaredNorm();
            const double weight = weights[k] * loss_fn.weight(r_squared);

            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // differentiate residual with respect to z
            Eigen::Matrix2d dr_dz = (zh * x[k].transpose() + alpha * Eigen::Matrix2d::Identity()) *
                                    (Eigen::Matrix2d::Identity() - zh * zh.transpose()) / n_z;

            Eigen::Matrix<double, 2, 5> dz;
            dz << 0.0, RX(2), -RX(1), 1.0, 0.0, -RX(2), 0.0, RX(0), 0.0, 1.0;

            Eigen::Matrix<double, 2, 5> J = dr_dz * dz;

            // Accumulate into JtJ and Jtr
            Jtr += weight * J.transpose() * r;
            for (size_t i = 0; i < 5; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J.col(i).dot(J.col(j)));
                }
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 5, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_pre(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t(0) = pose.t(0) + dp(3);
        pose_new.t(1) = pose.t(1) + dp(4);
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 5;

  private:
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

} // namespace poselib

#endif
