// From Practical solutions to the relative pose of three calibrated cameras, CVPR 25
// From poselib implementation

#include <iostream>
#include <Eigen/Eigen>
#include "helpers_eigen.h"




typedef Eigen::Vector2d Point2D;
typedef Eigen::Vector3d Point3D;

struct BundleStats {
    size_t iterations = 0;
    double initial_cost;
    double cost;
    double lambda;
    size_t invalid_steps;
    double step_norm;
    double grad_norm;
};

struct BundleOptions {
    size_t max_iterations = 100;
    enum LossType {
        TRIVIAL,
        TRUNCATED,
        HUBER,
        CAUCHY,
        // This is the TR-IRLS scheme from Le and Zach, 3DV 2021
        TRUNCATED_LE_ZACH
    } loss_type = LossType::CAUCHY;
    double loss_scale = 1.0;
    double gradient_tol = 1e-10;
    double step_tol = 1e-8;
    double initial_lambda = 1e-3;
    double min_lambda = 1e-10;
    double max_lambda = 1e10;
    bool verbose = false;
};

#define SWITCH_LOSS_FUNCTIONS                                                                                          \
    case BundleOptions::LossType::TRIVIAL:                                                                             \
        SWITCH_LOSS_FUNCTION_CASE(TrivialLoss);                                                                        \
        break;                                                                                                         \
    case BundleOptions::LossType::TRUNCATED:                                                                           \
        SWITCH_LOSS_FUNCTION_CASE(TruncatedLoss);                                                                      \
        break;                                                                                                         \
    case BundleOptions::LossType::HUBER:                                                                               \
        SWITCH_LOSS_FUNCTION_CASE(HuberLoss);                                                                          \
        break;                                                                                                         \
    case BundleOptions::LossType::CAUCHY:                                                                              \
        SWITCH_LOSS_FUNCTION_CASE(CauchyLoss);                                                                         \
        break;                                                                                                         \
    case BundleOptions::LossType::TRUNCATED_LE_ZACH:                                                                   \
        SWITCH_LOSS_FUNCTION_CASE(TruncatedLossLeZach);                                                                \
        break;

class UniformWeightVector {
public:
    UniformWeightVector() {}
    constexpr double operator[](std::size_t idx) const { return 1.0; }
};


inline Eigen::Matrix3d quat_to_rotmat(const Eigen::Vector4d& q) {
    return Eigen::Quaterniond(q(0), q(1), q(2), q(3)).toRotationMatrix();
}

inline Eigen::Vector4d rotmat_to_quat(const Eigen::Matrix3d& R) {
    Eigen::Quaterniond q_flip(R);
    Eigen::Vector4d q;
    q << q_flip.w(), q_flip.x(), q_flip.y(), q_flip.z();
    q.normalize();
    return q;
}

inline Eigen::Vector3d quat_rotate(const Eigen::Vector4d& q, const Eigen::Vector3d& p) {
    const double q1 = q(0), q2 = q(1), q3 = q(2), q4 = q(3);
    const double p1 = p(0), p2 = p(1), p3 = p(2);
    const double px1 = -p1 * q2 - p2 * q3 - p3 * q4;
    const double px2 = p1 * q1 - p2 * q4 + p3 * q3;
    const double px3 = p2 * q1 + p1 * q4 - p3 * q2;
    const double px4 = p2 * q2 - p1 * q3 + p3 * q1;
    return Eigen::Vector3d(px2 * q1 - px1 * q2 - px3 * q4 + px4 * q3, px3 * q1 - px1 * q3 + px2 * q4 - px4 * q2,
        px3 * q2 - px2 * q3 - px1 * q4 + px4 * q1);
}
inline Eigen::Vector4d quat_conj(const Eigen::Vector4d& q) { return Eigen::Vector4d(q(0), -q(1), -q(2), -q(3)); }

inline Eigen::Vector4d quat_multiply(const Eigen::Vector4d& qa, const Eigen::Vector4d& qb) {
    const double qa1 = qa(0), qa2 = qa(1), qa3 = qa(2), qa4 = qa(3);
    const double qb1 = qb(0), qb2 = qb(1), qb3 = qb(2), qb4 = qb(3);

    return Eigen::Vector4d(qa1 * qb1 - qa2 * qb2 - qa3 * qb3 - qa4 * qb4, qa1 * qb2 + qa2 * qb1 + qa3 * qb4 - qa4 * qb3,
        qa1 * qb3 + qa3 * qb1 - qa2 * qb4 + qa4 * qb2,
        qa1 * qb4 + qa2 * qb3 - qa3 * qb2 + qa4 * qb1);
}

inline Eigen::Vector4d quat_exp(const Eigen::Vector3d& w) {
    const double theta2 = w.squaredNorm();
    const double theta = std::sqrt(theta2);
    const double theta_half = 0.5 * theta;

    double re, im;
    if (theta > 1e-6) {
        re = std::cos(theta_half);
        im = std::sin(theta_half) / theta;
    }
    else {
        // we are close to zero, use taylor expansion to avoid problems
        // with zero divisors in sin(theta/2)/theta
        const double theta4 = theta2 * theta2;
        re = 1.0 - (1.0 / 8.0) * theta2 + (1.0 / 384.0) * theta4;
        im = 0.5 - (1.0 / 48.0) * theta2 + (1.0 / 3840.0) * theta4;

        // for the linearized part we re-normalize to ensure unit length
        // here s should be roughly 1.0 anyways, so no problem with zero div
        const double s = std::sqrt(re * re + im * im * theta2);
        re /= s;
        im /= s;
    }
    return Eigen::Vector4d(re, im * w(0), im * w(1), im * w(2));
}

inline Eigen::Vector4d quat_step_post(const Eigen::Vector4d& q, const Eigen::Vector3d& w_delta) {
    return quat_multiply(q, quat_exp(w_delta));
}

struct alignas(32) CameraPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Rotation is represented as a unit quaternion
        // with real part first, i.e. QW, QX, QY, QZ
        Eigen::Vector4d q;
    Eigen::Vector3d t;

    // Constructors (Defaults to identity camera)
    CameraPose() : q(1.0, 0.0, 0.0, 0.0), t(0.0, 0.0, 0.0) {}
    CameraPose(const Eigen::Vector4d& qq, const Eigen::Vector3d& tt) : q(qq), t(tt) {}
    CameraPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& tt) : q(rotmat_to_quat(R)), t(tt) {}

    // Helper functions
    inline Eigen::Matrix3d R() const { return quat_to_rotmat(q); }
    inline Eigen::Matrix<double, 3, 4> Rt() const {
        Eigen::Matrix<double, 3, 4> tmp;
        tmp.block<3, 3>(0, 0) = quat_to_rotmat(q);
        tmp.col(3) = t;
        return tmp;
    }
    inline Eigen::Vector3d rotate(const Eigen::Vector3d& p) const { return quat_rotate(q, p); }
    inline Eigen::Vector3d derotate(const Eigen::Vector3d& p) const { return quat_rotate(quat_conj(q), p); }
    inline Eigen::Vector3d apply(const Eigen::Vector3d& p) const { return rotate(p) + t; }

    inline Eigen::Vector3d center() const { return -derotate(t); }
};

struct alignas(32) ThreeViewCameraPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Rotation is represented as a unit quaternion
        // with real part first, i.e. QW, QX, QY, QZ
    CameraPose pose12;
    CameraPose pose13;

    // Constructors (Defaults to identity camera)
    ThreeViewCameraPose() : pose12(CameraPose()), pose13(CameraPose()) {}
    ThreeViewCameraPose(CameraPose pose12, CameraPose pose13) : pose12(pose12), pose13(pose13) {}

    const CameraPose pose23() const {
        Eigen::Matrix3d R23 = pose13.R() * pose12.R().transpose();
        Eigen::Vector3d t23 = -R23 * pose12.t + pose13.t;

        return CameraPose(R23, t23);
    }
};

struct ThreeViews {
    Eigen::Matrix3d R2, R3;
    Eigen::Vector3d t2, t3;

    Eigen::Matrix<double, 3, 4> P1() const {
        Eigen::Matrix<double, 3, 4> P;
        P.block<3, 3>(0, 0).setIdentity();
        P.block<3, 1>(0, 3).setZero();
        return P;
    }

    Eigen::Matrix<double, 3, 4> P2() const {
        Eigen::Matrix<double, 3, 4> P;
        P.block<3, 3>(0, 0) = R2;
        P.block<3, 1>(0, 3) = t2;
        return P;
    }

    Eigen::Matrix<double, 3, 4> P3() const {
        Eigen::Matrix<double, 3, 4> P;
        P.block<3, 3>(0, 0) = R3;
        P.block<3, 1>(0, 3) = t3;
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
typedef std::function<void(const BundleStats& stats)> IterationCallback;

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

void essential_from_motion(const CameraPose& pose, Eigen::Matrix3d* E) {
    *E << 0.0, -pose.t(2), pose.t(1), pose.t(2), 0.0, -pose.t(0), -pose.t(1), pose.t(0), 0.0;
    *E = (*E) * pose.R();
}

Eigen::Matrix3d skew(const Eigen::Vector3d& x) {
    Eigen::Matrix3d s;
    s << 0, -x(2), x(1), x(2), 0, -x(0),
        -x(1), x(0), 0;
    return s;
}


template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class ThreeViewRelativePoseJacobianAccumulator {
public:
    ThreeViewRelativePoseJacobianAccumulator(const std::vector<Point2D>& points2D_1,
        const std::vector<Point2D>& points2D_2,
        const std::vector<Point2D>& points2D_3,
        const LossFunction& l,
        const ResidualWeightVector& w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), x3(points2D_3), loss_fn(l), weights(w) {}

    double residual(const ThreeViewCameraPose& three_view_pose) const {
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

    size_t accumulate(const ThreeViewCameraPose& three_view_pose, Eigen::Matrix<double, 11, 11>& JtJ,
        Eigen::Matrix<double, 11, 1>& Jtr) {
        // We use tangent bases for t12 and direct vector for t23
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)

        CameraPose pose12 = three_view_pose.pose12;

        if (std::abs(pose12.t.x()) < std::abs(pose12.t.y())) {
            // x < y
            if (std::abs(pose12.t.x()) < std::abs(pose12.t.z())) {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitX()).normalized();
            }
            else {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        else {
            // x > y
            if (std::abs(pose12.t.y()) < std::abs(pose12.t.z())) {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitY()).normalized();
            }
            else {
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
        dE13dt13_1.row(2) = -R13.row(0);

        dE13dt13_2.row(0) = -R13.row(1);
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
        dE23dr13_0 = -dE23dr12_0;
        dE23dr13_1 = -dE23dr12_1;
        dE23dr13_2 = -dE23dr12_2;

        // dE23dt12 = skew(tangent_basis_k) * R23
        Eigen::Matrix3d dE23dt12_0, dE23dt12_1;
        dE23dt12_0 = -skew(R23 * tangent_basis.col(0)) * R23;
        dE23dt12_1 = -skew(R23 * tangent_basis.col(1)) * R23;

        // dE23dt13 = skew(e_k) * R23
        Eigen::Matrix3d dE23dt13_0, dE23dt13_1, dE23dt13_2;
        dE23dt13_0.row(0).setZero();
        dE23dt13_0.row(1) = -R23.row(2);
        dE23dt13_0.row(2) = R23.row(1);

        dE23dt13_1.row(0) = R23.row(2);
        dE23dt13_1.row(1).setZero();
        dE23dt13_1.row(2) = -R23.row(0);

        dE23dt13_2.row(0) = -R23.row(1);
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
            J_C12 << E12.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), E12.block<2, 3>(0, 0)* x1[k].homogeneous();
            const double nJ_C12 = J_C12.norm();
            const double inv_nJ_C12 = 1.0 / nJ_C12;
            const double r12 = C12 * inv_nJ_C12;
            const double weight12 = weights[k] * loss_fn.weight(r12 * r12);
            if (weight12 == 0.0) {
                continue;
            }

            Eigen::Vector4d J_C13;
            J_C13 << E13.block<3, 2>(0, 0).transpose() * x3[k].homogeneous(), E13.block<2, 3>(0, 0)* x1[k].homogeneous();
            const double nJ_C13 = J_C13.norm();
            const double inv_nJ_C13 = 1.0 / nJ_C13;
            const double r13 = C13 * inv_nJ_C13;

            const double weight13 = weights[k] * loss_fn.weight(r13 * r13);
            if (weight13 == 0.0) {
                continue;
            }

            Eigen::Vector4d J_C23;
            J_C23 << E23.block<3, 2>(0, 0).transpose() * x3[k].homogeneous(), E23.block<2, 3>(0, 0)* x2[k].homogeneous();
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
            dSdE12 << x1[k](0) * x2[k](0), x1[k](0)* x2[k](1), x1[k](0), x1[k](1)* x2[k](0), x1[k](1)* x2[k](1),
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
            dSdE13 << x1[k](0) * x3[k](0), x1[k](0)* x3[k](1), x1[k](0), x1[k](1)* x3[k](0), x1[k](1)* x3[k](1),
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
            dSdE23 << x2[k](0) * x3[k](0), x2[k](0)* x3[k](1), x2[k](0), x2[k](1)* x3[k](0), x2[k](1)* x3[k](1),
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

    ThreeViewCameraPose step(Eigen::Matrix<double, 11, 1> dp, const ThreeViewCameraPose& three_view_pose) const {
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
    const std::vector<Point2D>& x1;
    const std::vector<Point2D>& x2;
    const std::vector<Point2D>& x3;
    const LossFunction& loss_fn;
    const ResidualWeightVector& weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

template <typename Problem, typename Param = typename Problem::param_t>
BundleStats lm_impl(Problem& problem, Param* parameters, const BundleOptions& opt,
    IterationCallback callback = nullptr) {
    constexpr int n_params = Problem::num_params;
    Eigen::Matrix<double, n_params, n_params> JtJ;
    Eigen::Matrix<double, n_params, 1> Jtr;

    // Initialize
    BundleStats stats;
    stats.cost = problem.residual(*parameters);
    stats.initial_cost = stats.cost;
    stats.grad_norm = -1;
    stats.step_norm = -1;
    stats.invalid_steps = 0;
    stats.lambda = opt.initial_lambda;

    bool recompute_jac = true;
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; ++stats.iterations) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            problem.accumulate(*parameters, JtJ, Jtr);
            stats.grad_norm = Jtr.norm();
            if (stats.grad_norm < opt.gradient_tol) {
                break;
            }
        }

        // Add dampening
        for (size_t k = 0; k < n_params; ++k) {
            JtJ(k, k) += stats.lambda;
        }

        Eigen::Matrix<double, n_params, 1> sol = -JtJ.template selfadjointView<Eigen::Lower>().llt().solve(Jtr);

        stats.step_norm = sol.norm();
        if (stats.step_norm < opt.step_tol) {
            break;
        }

        Param parameters_new = problem.step(sol, *parameters);

        double cost_new = problem.residual(parameters_new);

        if (cost_new < stats.cost) {
            *parameters = parameters_new;
            stats.lambda = std::max(opt.min_lambda, stats.lambda / 10);
            stats.cost = cost_new;
            recompute_jac = true;
        }
        else {
            stats.invalid_steps++;
            // Remove dampening
            for (size_t k = 0; k < n_params; ++k) {
                JtJ(k, k) -= stats.lambda;
            }
            stats.lambda = std::min(opt.max_lambda, stats.lambda * 10);
            recompute_jac = false;
        }
        if (callback != nullptr) {
            callback(stats);
        }
    }
    return stats;
}

// Robust loss functions
class TrivialLoss {
public:
    TrivialLoss(double) {} // dummy to ensure we have consistent calling interface
    TrivialLoss() {}
    double loss(double r2) const { return r2; }
    double weight(double r2) const { return 1.0; }
};

class TruncatedLoss {
public:
    TruncatedLoss(double threshold) : squared_thr(threshold* threshold) {}
    double loss(double r2) const { return std::min(r2, squared_thr); }
    double weight(double r2) const { return (r2 < squared_thr) ? 1.0 : 0.0; }

private:
    const double squared_thr;
};

// The method from
//  Le and Zach, Robust Fitting with Truncated Least Squares: A Bilevel Optimization Approach, 3DV 2021
// for truncated least squares optimization with IRLS.
class TruncatedLossLeZach {
public:
    TruncatedLossLeZach(double threshold) : squared_thr(threshold* threshold), mu(0.5) {}
    double loss(double r2) const { return std::min(r2, squared_thr); }
    double weight(double r2) const {
        double r2_hat = r2 / squared_thr;
        double zstar = std::min(r2_hat, 1.0);

        if (r2_hat < 1.0) {
            return 0.5;
        }
        else {
            // assumes mu > 0.5
            double r2m1 = r2_hat - 1.0;
            double rho = (2.0 * r2m1 + std::sqrt(4.0 * r2m1 * r2m1 * mu * mu + 2 * mu * r2m1)) / mu;
            double a = (r2_hat + mu * rho * zstar - 0.5 * rho) / (1 + mu * rho);
            double zbar = std::max(0.0, std::min(a, 1.0));
            return (zstar - zbar) / rho;
        }
    }

private:
    const double squared_thr;

public:
    // hyper-parameter for penalty strength
    double mu;
    // schedule for increasing mu in each iteration
    static constexpr double alpha = 1.5;
};

class HuberLoss {
public:
    HuberLoss(double threshold) : thr(threshold) {}
    double loss(double r2) const {
        const double r = std::sqrt(r2);
        if (r <= thr) {
            return r2;
        }
        else {
            return thr * (2.0 * r - thr);
        }
    }
    double weight(double r2) const {
        const double r = std::sqrt(r2);
        if (r <= thr) {
            return 1.0;
        }
        else {
            return thr / r;
        }
    }

private:
    const double thr;
};
class CauchyLoss {
public:
    CauchyLoss(double threshold) : inv_sq_thr(1.0 / (threshold * threshold)) {}
    double loss(double r2) const { return std::log1p(r2 * inv_sq_thr); }
    double weight(double r2) const {
        return std::max(std::numeric_limits<double>::min(), 1.0 / (1.0 + r2 * inv_sq_thr));
    }

private:
    const double inv_sq_thr;
};

// Callback which prints debug info from the iterations
void print_iteration(const BundleStats& stats) {
    if (stats.iterations == 0) {
        std::cout << "initial_cost=" << stats.initial_cost << "\n";
    }
    std::cout << "iter=" << stats.iterations << ", cost=" << stats.cost << ", step=" << stats.step_norm
        << ", grad=" << stats.grad_norm << ", lambda=" << stats.lambda << "\n";
}

template <typename LossFunction> IterationCallback setup_callback(const BundleOptions& opt, LossFunction& loss_fn) {
    if (opt.verbose) {
        return print_iteration;
    }
    else {
        return nullptr;
    }
}

// For using the IRLS scheme proposed by Le and Zach 3DV2021, we have a callback
// for each iteration which updates the mu parameter
template <> IterationCallback setup_callback(const BundleOptions& opt, TruncatedLossLeZach& loss_fn) {
    if (opt.verbose) {
        return [&loss_fn](const BundleStats& stats) {
            print_iteration(stats);
            loss_fn.mu *= TruncatedLossLeZach::alpha;
        };
    }
    else {
        return [&loss_fn](const BundleStats& stats) { loss_fn.mu *= TruncatedLossLeZach::alpha; };
    }
}

template <typename WeightType, typename LossFunction>
BundleStats refine_3v_relpose(const std::vector<Point2D>& x1, const std::vector<Point2D>& x2,
    const std::vector<Point2D>& x3, ThreeViewCameraPose* pose,
    const BundleOptions& opt, const WeightType& weights) {
    LossFunction loss_fn(opt.loss_scale);
    IterationCallback callback = setup_callback(opt, loss_fn);
    ThreeViewRelativePoseJacobianAccumulator<LossFunction, WeightType> accum(x1, x2, x3, loss_fn, weights);
    return lm_impl<decltype(accum)>(accum, pose, opt, callback);
}

template <typename WeightType>
BundleStats refine_3v_relpose1(const std::vector<Point2D>& x1, const std::vector<Point2D>& x2,
    const std::vector<Point2D>& x3, ThreeViewCameraPose* pose,
    const BundleOptions& opt, const WeightType& weights) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return refine_3v_relpose<WeightType, LossFunction>(x1, x2, x3, pose, opt, weights);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for essential matrix refinement
BundleStats refine_3v_relpose(const std::vector<Point2D>& x1, const std::vector<Point2D>& x2,
    const std::vector<Point2D>& x3, ThreeViewCameraPose* pose,
    const BundleOptions& opt, const std::vector<double>& weights) {
    if (weights.size() == x1.size()) {
        return refine_3v_relpose1<std::vector<double>>(x1, x2, x3, pose, opt, weights);
    }
    else {
        return refine_3v_relpose1<UniformWeightVector>(x1, x2, x3, pose, opt, UniformWeightVector());
    }
}



int solver_4p3v(const Eigen::Matrix<double, 2, 4>& x1,
    const Eigen::Matrix<double, 2, 4>& x2,
    const Eigen::Matrix<double, 2, 4>& x3,
    ThreeViewsVector* solutions, int iters);

double solver_4p3v_para(const Eigen::Matrix<double, 2, 4>& x1,
    const Eigen::Matrix<double, 2, 4>& x2,
    const Eigen::Matrix<double, 2, 4>& x3,
    Solution* solution);

void centering_rotation(const Eigen::Vector3d& x0, Eigen::Matrix3d* R);

void solve_for_translation(const Eigen::Matrix<double, 2, 4>& x1, const Eigen::Matrix<double, 2, 4>& x2,
    const Eigen::Matrix<double, 2, 4>& x3, const Eigen::Matrix3d& R2, const Eigen::Matrix3d& R3,
    Eigen::Matrix<double, 3, 1>* t2, Eigen::Matrix<double, 3, 1>* t3);

void solver_4p3v_para(const std::vector<Eigen::Vector2d>& x1, const std::vector<Eigen::Vector2d>& x2,
    const std::vector<Eigen::Vector2d>& x3, const std::vector<size_t>& sample,
    std::vector<ThreeViewCameraPose>* models, int iters = 100, double sq_epipolar_t = 1.0);




void weight_points(const Eigen::Matrix<double, 1, 3>& z,
    const Eigen::Matrix<double, 2, 4>& x, Eigen::Matrix<double, 2, 4>* xp)
{
    xp->col(0) = x.col(0);
    xp->col(1) = x.col(1) * (1 + z(0));
    xp->col(2) = x.col(2) * (1 + z(1));
    xp->col(3) = x.col(3) * (1 + z(2));
}

int solver_4p3v(const Eigen::Matrix<double, 2, 4>& x1,
    const Eigen::Matrix<double, 2, 4>& x2,
    const Eigen::Matrix<double, 2, 4>& x3,
    ThreeViewsVector* solutions, int iters)
{

    // Rotate so that first point is at the center
    Eigen::Matrix3d R1, R2, R3;
    centering_rotation(x1.col(0).homogeneous(), &R1);
    centering_rotation(x2.col(0).homogeneous(), &R2);
    centering_rotation(x3.col(0).homogeneous(), &R3);

    // Centered coordinates
    Eigen::Matrix<double, 2, 4> x1c, x2c, x3c;
    x1c = (R1 * x1.colwise().homogeneous()).colwise().hnormalized();
    x2c = (R2 * x2.colwise().homogeneous()).colwise().hnormalized();
    x3c = (R3 * x3.colwise().homogeneous()).colwise().hnormalized();

    Solution sol;

    // Relative depths
    Eigen::Matrix<double, 1, 3> z1, z2, z3;

    solver_4p3v_para(x1c, x2c, x3c, &sol);

    Eigen::Matrix<double, 2, 4> x1p, x2p, x3p;

    Solution sol1, sol2;
    for (int iter = 0; iter < iters - 1; ++iter) {
        // Weighted coordinates

        weight_points(sol.z1, x1c, &x1p);
        weight_points(sol.z2, x2c, &x2p);
        weight_points(sol.z3, x3c, &x3p);

        double score1 = solver_4p3v_para(x1p, x2p, x3p, &sol1);

        // Solve for flipped solution
        weight_points(-sol.z1, x1c, &x1p);
        weight_points(-sol.z2, x2c, &x2p);
        weight_points(-sol.z3, x3c, &x3p);

        double score2 = solver_4p3v_para(x1p, x2p, x3p, &sol2);

        //std::cout << "iter = " << iter << ", score = [" << score1 << ", " << score2 << "]\n";

        if (score1 < score2) {
            sol = sol1;
        }
        else {
            sol = sol2;
        }
    }

    solutions->clear();
    solutions->push_back(sol);
    solutions->push_back(sol.flipped());

    // Revert coordinate change
    for (ThreeViews& s : *solutions) {
        s.R2 = R2 * s.R2 * R1.transpose();
        s.R3 = R3 * s.R3 * R1.transpose();

        solve_for_translation(x1c, x2c, x3c, s.R2, s.R3, &s.t2, &s.t3);
    }


    return 0;
}

double solver_4p3v_para(const Eigen::Matrix<double, 2, 4>& x1,
    const Eigen::Matrix<double, 2, 4>& x2,
    const Eigen::Matrix<double, 2, 4>& x3,
    Solution* sol)
{

    Eigen::Matrix<double, 2, 3> A, B, C;
    A << x1.col(1) - x1.col(0), x1.col(2) - x1.col(0), x1.col(3) - x1.col(0);
    B << x2.col(1) - x2.col(0), x2.col(2) - x2.col(0), x2.col(3) - x2.col(0);
    C << x3.col(1) - x3.col(0), x3.col(2) - x3.col(0), x3.col(3) - x3.col(0);

    Eigen::Matrix<double, 4, 3> tmp;
    Eigen::Matrix<double, 4, 4> Q;
    Eigen::Matrix<double, 4, 1> ur, us, uv;

    tmp << A, B;
    Q = tmp.householderQr().householderQ();
    ur = Q.rightCols(1);

    tmp << C, A;
    Q = tmp.householderQr().householderQ();
    us = Q.rightCols(1);

    tmp << B, C;
    Q = tmp.householderQr().householderQ();
    uv = Q.rightCols(1);

    Eigen::Matrix<double, 2, 1> u, v, up, vp, upp, vpp;
    u << -ur(3), ur(2);
    v << -ur(1), ur(0);
    up << -us(3), us(2);
    vp << -us(1), us(0);
    upp << -uv(3), uv(2);
    vpp << -uv(1), uv(0);

    double rv = v(1) / v(0);
    double ru = u(1) / u(0);
    double ruv = u(0) / v(0);
    double rvu2 = v(1) / u(1);
    double rvp = vp(1) / vp(0);
    double rup = up(1) / up(0);
    double ruvp = up(0) / vp(0);
    double rvup2 = vp(1) / up(1);
    double rvpp = vpp(1) / vpp(0);
    double rupp = upp(1) / upp(0);
    double ruvpp = upp(0) / vpp(0);
    double rvupp2 = vpp(1) / upp(1);



    double T12 = std::sqrt((rv * rv + 1) / ((ru * ru + 1) * (ruv * ruv)));
    double T31 = std::sqrt((rvp * rvp + 1) / ((rup * rup + 1) * ruvp * ruvp));
    double T23 = std::sqrt((rvpp * rvpp + 1) / ((rupp * rupp + 1) * ruvpp * ruvpp));

    Eigen::Matrix<double, 4, 1> nr, ns, nv;
    nr << 1.0, ru, (1.0 / T12)* ru* rvu2 / rv, (1.0 / T12)* ru* rvu2;
    ns << 1.0, rup, (1.0 / T31)* rup* rvup2 / rvp, (1.0 / T31)* rup* rvup2;
    nv << 1.0, rupp, (1.0 / T23)* rupp* rvupp2 / rvpp, (1.0 / T23)* rupp* rvupp2;

    double pq1 = (nr(2) + nr(0)) / (nr(3) + nr(1));
    double rs1 = (nr(2) + nr(0)) / (nr(3) - nr(1));
    double pq2 = (ns(2) + ns(0)) / (ns(3) + ns(1));
    double rs2 = (ns(2) + ns(0)) / (ns(3) - ns(1));
    double pq3 = (nv(2) + nv(0)) / (nv(3) + nv(1));
    double rs3 = (nv(2) + nv(0)) / (nv(3) - nv(1));

    double A1 = std::atan(pq1);
    double B1 = std::atan(rs1);
    double A2 = std::atan(pq2);
    double B2 = std::atan(rs2);
    double A3 = std::atan(pq3);
    double B3 = std::atan(rs3);

    double S0 = std::sin(B1 + B2 + B3);
    double S1 = std::sin(B1 - A2 + A3);
    double S2 = std::sin(B2 - A3 + A1);
    double S3 = std::sin(B3 - A1 + A2);

    double tC1 = std::sqrt(std::abs((S0 * S1) / (S2 * S3)));


    //for(int i = 0; i < 2; ++i) {

    double C1 = std::atan(tC1);
    double C2 = std::atan(tC1 * (S2 / S1));
    //double C3 = std::atan(tC1*(S3/S1));

    Eigen::Quaternion<double> qr, qs, qv;
    qr.coeffs() << std::sin(A1) * std::sin(C1),
        std::cos(A1)* std::sin(C1),
        std::sin(B1)* std::cos(C1),
        std::cos(B1)* std::cos(C1);



    qs.coeffs() << std::sin(A2) * std::sin(C2),
        std::cos(A2)* std::sin(C2),
        std::sin(B2)* std::cos(C2),
        -std::cos(B2) * std::cos(C2);

    /*
            qv.coeffs() << std::sin(A3)*std::sin(C3),
                           std::cos(A3)*std::sin(C3),
                           std::sin(B3)*std::cos(C3),
                           std::cos(B3)*std::cos(C3);
      */

    Eigen::Matrix<double, 3, 3> R, S;
    R = qr.toRotationMatrix();
    S = qs.toRotationMatrix();

    //Solution sol;
    sol->R2 = R;
    sol->R3 = S;

    sol->T12 = T12;
    sol->T23 = T23;
    sol->T31 = T31;

    //solutions->emplace_back(sol);
    sol->z1 = ((1.0 / T12) * R.block<2, 1>(0, 2).transpose() * B + R(2, 2) * R.block<1, 2>(2, 0) * A) / (1 - R(2, 2) * R(2, 2));
    sol->z2 = T12 * (R.block<1, 2>(2, 0) * x1.block<2, 3>(0, 1) + R(2, 2) * sol->z1);
    sol->z3 = T23 * ((S.row(2) * (R.block<2, 3>(0, 0).transpose())) * x2.block<2, 3>(0, 1) + (S.row(2) * R.row(2).transpose()) * sol->z2);


    //tC1 = -tC1;
    //}

    return std::abs(1.0 - T12 * T23 * T31);
}


void centering_rotation(const Eigen::Vector3d& x0, Eigen::Matrix3d* R)
{

    Eigen::Vector3d r3 = x0.normalized();
    Eigen::Vector3d r1{ -r3(2),0.0,r3(0) };
    r1.normalize();
    Eigen::Vector3d r2 = r3.cross(r1);

    R->row(0) = r1;
    R->row(1) = r2;
    R->row(2) = r3;
}



void solve_for_translation(const Eigen::Matrix<double, 2, 4>& x1, const Eigen::Matrix<double, 2, 4>& x2,
    const Eigen::Matrix<double, 2, 4>& x3, const Eigen::Matrix3d& R2, const Eigen::Matrix3d& R3,
    Eigen::Matrix<double, 3, 1>* t2, Eigen::Matrix<double, 3, 1>* t3)
{

    Eigen::Matrix<double, 3, 4> X1 = x1.colwise().homogeneous();
    Eigen::Matrix<double, 3, 4> X2 = x2.colwise().homogeneous();
    Eigen::Matrix<double, 3, 4> X3 = x3.colwise().homogeneous();

    Eigen::Matrix<double, 12, 6> A;
    A.setZero();

    for (int i = 0; i < 4; ++i) {
        A.block<1, 3>(i, 0) = (R2 * X1.col(i)).cross(X2.col(i));
        A.block<1, 3>(4 + i, 3) = (R3 * X1.col(i)).cross(X3.col(i));

        A.block<1, 3>(8 + i, 0) = -X2.col(i).cross(R2 * R3.transpose() * X3.col(i));
        A.block<1, 3>(8 + i, 3) = (R3 * R2.transpose() * X2.col(i)).cross(X3.col(i));
    }

    // Estimate translation directions from epipolar constraints
    Eigen::JacobiSVD<Eigen::Matrix<double, 12, 6>> svd(A, Eigen::ComputeFullV);
    Eigen::Matrix<double, 6, 1> tt = svd.matrixV().col(5);
    tt = tt / tt.block<3, 1>(0, 0).norm();

    *t2 = tt.block<3, 1>(0, 0);
    *t3 = tt.block<3, 1>(3, 0);

    // Correct sign using chirality of first point
    double n2 = x2.col(0).squaredNorm();
    Eigen::Vector3d X = R2 * x1.col(0).homogeneous();
    double lambda = (x2.col(0).dot(t2->block<2, 1>(0, 0)) - (*t2)(2) * n2) / (n2 * X(2) - x2.col(0).dot(X.block<2, 1>(0, 0)));
    if (lambda < 0) {
        *t2 *= -1.0;
        *t3 *= -1.0;
    }

}

// ENTRY???
void solver_4p3v_para
(
    Eigen::Matrix<double, 2, 4> const& xx1, //const std::vector<Eigen::Vector2d>& x1,
    Eigen::Matrix<double, 2, 4> const& xx2,  //const std::vector<Eigen::Vector2d>& x2,
    Eigen::Matrix<double, 2, 4> const& xx3, //const std::vector<Eigen::Vector2d>& x3,
    //const std::vector<size_t>& sample,
    std::vector<ThreeViewCameraPose>& models,
    int iters, // 0 default
    double sq_epipolar_t
)
{
    //Eigen::Matrix<double, 2, 4> xx1, xx2, xx3;

    //for (int k = 0; k < 4; ++k) 
    //{
    //    xx1(0, k) = x1[sample[k]][0];
    //    xx1(1, k) = x1[sample[k]][1];

    //    xx2(0, k) = x2[sample[k]][0];
    //    xx2(1, k) = x2[sample[k]][1];

    //    xx3(0, k) = x3[sample[k]][0];
    //    xx3(1, k) = x3[sample[k]][1];
    //}

    ThreeViewsVector solutions;

    solver_4p3v(xx1, xx2, xx3, &solutions, iters);

    models.reserve(solutions.size());

    for (ThreeViews sol : solutions) 
    {
        Eigen::Matrix3d E12, E13;
        essential_from_motion(CameraPose(sol.R2, sol.t2), &E12);
        essential_from_motion(CameraPose(sol.R3, sol.t3), &E13);

        //std::cout << "R2: " << std::endl << sol.R2 << std::endl;
        //std::cout << "t2: " << sol.t2.transpose() << std::endl;
        //std::cout << "R3:" << std::endl << sol.R3 << std::endl;
        //std::cout << "t3: " << sol.t3.transpose() << std::endl;

        //std::vector<char> inliers_12, inliers_13;
        //int num_inliers_12 = get_inliers(E12, x1, x2, sq_epipolar_t, &inliers_12);
        //int num_inliers_13 = get_inliers(E13, x1, x3, sq_epipolar_t, &inliers_13);

        //std::cout << "Num inliers 12: " << num_inliers_12 << " Num inliers 13: " << num_inliers_13 << std::endl;

        models.emplace_back(ThreeViewCameraPose(CameraPose(sol.R2, sol.t2), CameraPose(sol.R3, sol.t3)));
    }
}





bool
solver_4p3v_para
(
    float const* p2d_1,
    float const* p2d_2,
    float const* p2d_3,
    float const* p3d_1,
    bool use_prior,
    int N,
    float* r1,
    float* t1,
    float* r2,
    float* t2
)
{
    Eigen::Matrix<double, 2, 4> p1 = matrix_from_buffer<float, 2, 4>(p2d_1).cast<double>();
    Eigen::Matrix<double, 2, 4> p2 = matrix_from_buffer<float, 2, 4>(p2d_2).cast<double>();
    Eigen::Matrix<double, 2, 4> p3 = matrix_from_buffer<float, 2, 4>(p2d_3).cast<double>();

    std::vector<ThreeViewCameraPose> sols;

    solver_4p3v_para(p1, p2, p3, sols, 2, 0);

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = 10;
    bundle_opt.max_iterations = 2;

    std::vector<Point2D> x1;
    std::vector<Point2D> x2;
    std::vector<Point2D> x3;

    for (int i = 0; i < N; ++i)
    {
        x1.push_back(Point2D(p2d_1[2 * i + 0], p2d_1[2 * i + 1]));
        x2.push_back(Point2D(p2d_2[2 * i + 0], p2d_2[2 * i + 1]));
        x3.push_back(Point2D(p2d_3[2 * i + 0], p2d_3[2 * i + 1]));
    }

    for (int i = 0; i < sols.size(); ++i)
    {
        ThreeViewCameraPose* sol = &sols[i];

        refine_3v_relpose(x1, x2, x3, sol, bundle_opt, std::vector<double>());

        std::cout << "pose12" << std::endl;
        std::cout << sol->pose12.R() << std::endl;
        std::cout << sol->pose12.t << std::endl;
        std::cout << "pose23" << std::endl;
        std::cout << sol->pose23().R() << std::endl;
        std::cout << sol->pose23().t << std::endl;


        //std::cout << "R2: " << std::endl << sol->R2 << std::endl;
        //std::cout << "t2: " << sol->t2.transpose() << std::endl;
        //std::cout << "R3:" << std::endl << sol->R3 << std::endl;
        //std::cout << "t3: " << sol.t3.transpose() << std::endl;
    }


    return true;
}
