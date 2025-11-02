//
// Created by kocur on 10-Nov-24.
//

#include "threeview_nister.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/utils.h"
#include "p3p.h"
#include "relpose_5pt.h"
#include <iostream>

namespace poselib {
namespace nister {

Eigen::MatrixXd Companion_Matrix(Eigen::VectorXd C) {

    Eigen::MatrixXd Cp(4, 4);
    Cp << 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -C(4), -C(3), -C(2), -C(1);
    return Cp;
}

Eigen::VectorXd Simplify_Poly_Coeff(Eigen::VectorXd C) {
    /*
    Dividing the coefficients so that the 1st will be equal to 1
    */
    Eigen::VectorXd C1(5);
    C1 << 1, C(1) / C(0), C(2) / C(0), C(3) / C(0), C(4) / C(0);
    return C1;
}

Eigen::VectorXd Quartic_Poly_Coeff(double a1, double b1, double c1, double d1, double e1, double f1, double a2,
                                   double b2, double c2, double d2, double e2, double f2) {
    /*
				**Input: The coefficients of the quadradic representations of the two conics B and G
				**Output: Coefficients of the quartic polynomial, of which the roots coincide with the intersection between two conic *Two steps of Gauss-Jordan *Derived using symbolic computations in MATLAB *QPC(0) is the coefficient of the degree 4, and so on
     */
    Eigen::VectorXd QPC(5);
    QPC(0) = (pow((a1 * c2 - a2 * c1), 2) / pow((a1 * b2 - a2 * b1), 2) - (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1));
    QPC(1) = ((2 * (a1 * c2 - a2 * c1) * (a1 * e2 - a2 * e1)) / pow((a1 * b2 - a2 * b1), 2) -
              (b1 * e2 - b2 * e1) / (a1 * b2 - a2 * b1) +
              ((a1 * c2 - a2 * c1) * (b1 * d2 - b2 * d1)) / pow((a1 * b2 - a2 * b1), 2) -
              (2 * (a1 * d2 - a2 * d1) * (b1 * c2 - b2 * c1)) / pow((a1 * b2 - a2 * b1), 2));
    QPC(2) =
        ((pow((a1 * e2 - a2 * e1), 2) + 2 * (a1 * c2 - a2 * c1) * (a1 * f2 - a2 * f1)) / pow((a1 * b2 - a2 * b1), 2) -
         (b1 * f2 - b2 * f1) / (a1 * b2 - a2 * b1) -
         (pow((a1 * d2 - a2 * d1), 2) * (b1 * c2 - b2 * c1)) / pow((a1 * b2 - a2 * b1), 3) -
         (2 * (a1 * d2 - a2 * d1) * (b1 * e2 - b2 * e1)) / pow((a1 * b2 - a2 * b1), 2) +
         ((a1 * e2 - a2 * e1) * (b1 * d2 - b2 * d1)) / pow((a1 * b2 - a2 * b1), 2) +
         ((a1 * c2 - a2 * c1) * (a1 * d2 - a2 * d1) * (b1 * d2 - b2 * d1)) / pow((a1 * b2 - a2 * b1), 3));
    QPC(3) = ((2 * (a1 * e2 - a2 * e1) * (a1 * f2 - a2 * f1)) / pow((a1 * b2 - a2 * b1), 2) -
              (2 * (a1 * d2 - a2 * d1) * (b1 * f2 - b2 * f1)) / pow((a1 * b2 - a2 * b1), 2) -
              (pow((a1 * d2 - a2 * d1), 2) * (b1 * e2 - b2 * e1)) / pow((a1 * b2 - a2 * b1), 3) +
              ((b1 * d2 - b2 * d1) * (a1 * f2 - a2 * f1)) / pow((a1 * b2 - a2 * b1), 2) +
              ((a1 * d2 - a2 * d1) * (a1 * e2 - a2 * e1) * (b1 * d2 - b2 * d1)) / pow((a1 * b2 - a2 * b1), 3));
    QPC(4) = pow((a1 * f2 - a2 * f1), 2) / pow((a1 * b2 - a2 * b1), 2) -
             (pow((a1 * d2 - a2 * d1), 2) * (b1 * f2 - b2 * f1)) / pow((a1 * b2 - a2 * b1), 3) +
             ((a1 * d2 - a2 * d1) * (b1 * d2 - b2 * d1) * (a1 * f2 - a2 * f1)) / pow((a1 * b2 - a2 * b1), 3);
    return QPC;
}

Eigen::Matrix3d essential_matrix_conic_1(double y1, double y2, double y3, double y4, double y5, double y6, double z1,
                                         double z2, double z3, double z4, double z5, double z6, double w1, double w2,
                                         double w3, double w4, double w5, double w6) {
    Eigen::Matrix3d X;
    double A, B, C, D, E, F;
    A = (y1 * y2 + y3 * y4 + y5 * y6);
    B = (y1 * z2 + y2 * z1 + y3 * z4 + y4 * z3 + y5 * z6 + y6 * z5);
    C = (z1 * z2 + z3 * z4 + z5 * z6);
    D = (w1 * y2 + w2 * y1 + w3 * y4 + w4 * y3 + w5 * y6 + w6 * y5);
    E = (w1 * z2 + w2 * z1 + w3 * z4 + w4 * z3 + w5 * z6 + w6 * z5);
    F = (w1 * w2 + w3 * w4 + w5 * w6);

    X << A, B / 2, D / 2, B / 2, C, E / 2, D / 2, E / 2, F;

    return X;
}

double y_coord_from_z(double a1, double b1, double c1, double d1, double e1, double f1, double a2, double b2, double c2,
                      double d2, double e2, double f2, double z) {
    /*
				**Uses an equation from Quartic_Poly_Coeff function
     */
    double y;
    y = -(a1 * f2 - a2 * f1 + a1 * e2 * z - a2 * e1 * z + a1 * c2 * pow(z, 2) - a2 * c1 * pow(z, 2)) /
        (a1 * d2 - a2 * d1 + a1 * b2 * z - a2 * b1 * z);
    return y;
}

Eigen::Matrix3d essential_matrix_conic_2(double y1, double y2, double y3, double y4, double y5, double y6, double z1,
                                         double z2, double z3, double z4, double z5, double z6, double w1, double w2,
                                         double w3, double w4, double w5, double w6) {
    Eigen::Matrix3d X;
    double A, B, C, D, E, F;
    A = (pow(y1, 2) - pow(y2, 2) + pow(y3, 2) - pow(y4, 2) + pow(y5, 2) - pow(y6, 2));
    B = (2 * y1 * z1 - 2 * y2 * z2 + 2 * y3 * z3 - 2 * y4 * z4 + 2 * y5 * z5 - 2 * y6 * z6);
    C = (pow(z1, 2) - pow(z2, 2) + pow(z3, 2) - pow(z4, 2) + pow(z5, 2) - pow(z6, 2));
    D = (2 * w1 * y1 - 2 * w2 * y2 + 2 * w3 * y3 - 2 * w4 * y4 + 2 * w5 * y5 - 2 * w6 * y6);
    E = (2 * w1 * z1 - 2 * w2 * z2 + 2 * w3 * z3 - 2 * w4 * z4 + 2 * w5 * z5 - 2 * w6 * z6);
    F = (pow(w1, 2) - pow(w2, 2) + pow(w3, 2) - pow(w4, 2) + pow(w5, 2) - pow(w6, 2));

    X << A, B / 2, D / 2, B / 2, C, E / 2, D / 2, E / 2, F;

    return X;
}

Eigen::Matrix3d *_3pt_to_Ematrix(Eigen::Vector3d *P, Eigen::Vector3d *Q, Eigen::Matrix3d R) {
    /*
				**Input: 3x Rotated points of view#1, 3x points in view#2, and the rotation matrix R
				**First we construct the 3x eq.46 vectors, one for each PC. Then computes the nullspace of this 3x6 matrix. Using this nullspace we compute *y1:6, z1:6, and w1:6 number, which are used to generate the conics (eq.49 and eq.50). Then, out of these conics we generate the quartic poly to find *the intersections, and the compute all the E1:6 numbers. Following eq.44, we construct the essential matrices, which are then multiplied from the right *side by the rotation matrix R.
     */
    Eigen::MatrixXd X_tilde(3, 6);
    Eigen::Matrix3d *Es;
    Es = (Eigen::Matrix3d *)malloc(4 * sizeof(Eigen::Matrix3d));
    X_tilde << Q[0](0) * P[0](0), Q[0](0) * P[0](1), Q[0](1) * P[0](0), Q[0](1) * P[0](1), Q[0](2) * P[0](0),
        Q[0](2) * P[0](1), Q[1](0) * P[1](0), Q[1](0) * P[1](1), Q[1](1) * P[1](0), Q[1](1) * P[1](1),
        Q[1](2) * P[1](0), Q[1](2) * P[1](1), Q[2](0) * P[2](0), Q[2](0) * P[2](1), Q[2](1) * P[2](0),
        Q[2](1) * P[2](1), Q[2](2) * P[2](0), Q[2](2) * P[2](1);
    // cout << X_tilde << endl;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(X_tilde);
    Eigen::MatrixXd A_null = lu.kernel();
    // cout << A_null << endl;

    double y1, y2, y3, y4, y5, y6, z1, z2, z3, z4, z5, z6, w1, w2, w3, w4, w5, w6;
    y1 = A_null.col(0)(0);
    y2 = A_null.col(0)(1);
    y3 = A_null.col(0)(2);
    y4 = A_null.col(0)(3);
    y5 = A_null.col(0)(4);
    y6 = A_null.col(0)(5);
    z1 = A_null.col(1)(0);
    z2 = A_null.col(1)(1);
    z3 = A_null.col(1)(2);
    z4 = A_null.col(1)(3);
    z5 = A_null.col(1)(4);
    z6 = A_null.col(1)(5);
    w1 = A_null.col(2)(0);
    w2 = A_null.col(2)(1);
    w3 = A_null.col(2)(2);
    w4 = A_null.col(2)(3);
    w5 = A_null.col(2)(4);
    w6 = A_null.col(2)(5);

    Eigen::Matrix3d C1, C2;
    C1 = essential_matrix_conic_1(y1, y2, y3, y4, y5, y6, z1, z2, z3, z4, z5, z6, w1, w2, w3, w4, w5, w6);
    C2 = essential_matrix_conic_2(y1, y2, y3, y4, y5, y6, z1, z2, z3, z4, z5, z6, w1, w2, w3, w4, w5, w6);

    Eigen::VectorXd QPC(5);

    QPC = Quartic_Poly_Coeff(C1(0, 0), 2 * C1(0, 1), C1(1, 1), 2 * C1(0, 2), 2 * C1(1, 2), C1(2, 2), C2(0, 0),
                             2 * C2(0, 1), C2(1, 1), 2 * C2(0, 2), 2 * C2(1, 2), C2(2, 2));
    QPC = Simplify_Poly_Coeff(QPC);
    // print_quartic_poly(QPC);
    Eigen::MatrixXd Cp(4, 4);
    Cp = Companion_Matrix(QPC);

    Eigen::EigenSolver<Eigen::MatrixXd> es(Cp);

    for (int i = 0; i < es.eigenvalues().size(); i++) {
        double E1, E2, E3, E4, E5, E6;
        Es[i] << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        // check only real solutions
        if (es.eigenvalues()(i).imag() == 0.0) {
            // calculate the y coordinate by backsubstituting the z
            double y, z = es.eigenvalues()(i).real();
            y = y_coord_from_z(C1(0, 0), 2 * C1(0, 1), C1(1, 1), 2 * C1(0, 2), 2 * C1(1, 2), C1(2, 2), C2(0, 0),
                               2 * C2(0, 1), C2(1, 1), 2 * C2(0, 2), 2 * C2(1, 2), C2(2, 2), z);

            E1 = y1 * y + z1 * z + w1;
            E2 = y2 * y + z2 * z + w2;
            E3 = y3 * y + z3 * z + w3;
            E4 = y4 * y + z4 * z + w4;
            E5 = y5 * y + z5 * z + w5;
            E6 = y6 * y + z6 * z + w6;

            Eigen::Matrix3d E;
            E << E1, E2, 0, E3, E4, 0, E5, E6, 0;
            Es[i] = E * R;
        }
    }
    return Es;
}

Eigen::Matrix3d DIAC(Eigen::Matrix3d K) {
    /*
				**Input: Matrix of Intrisic Camera parameters (upper triangular)
				**Output: Dual of the Image of the Absolute Conic
     */
    Eigen::Matrix3d omega_star;
    omega_star = K * K.transpose();
    return omega_star;
}

Eigen::Matrix3d calculateB(Eigen::Vector3d *p, double theta) {
    /*
				**Input: 4 points in a single view, and a parameter theta for the coordinates of the epipole
				**Output: The conic B passing from the points and the epipole with homogeneous coordinates (theta^2, theta, 1)
     */
    Eigen::Matrix3d B;
    Eigen::MatrixXd A(5, 6);
    Eigen::Vector3d e;

    e << pow(theta, 2), theta, 1;

    // A * [x^2 x*y y^2 x y 1]^T = 0
    A << pow(p[0](0), 2), p[0](1) * p[0](0), pow(p[0](1), 2), p[0](0), p[0](1), p[0](2), pow(p[1](0), 2),
        p[1](1) * p[1](0), pow(p[1](1), 2), p[1](0), p[1](1), p[1](2), pow(p[2](0), 2), p[2](1) * p[2](0),
        pow(p[2](1), 2), p[2](0), p[2](1), p[2](2), pow(p[3](0), 2), p[3](1) * p[3](0), pow(p[3](1), 2), p[3](0),
        p[3](1), p[3](2), pow(e(0), 2), e(1) * e(0), pow(e(1), 2), e(0), e(1), e(2);

    Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
    Eigen::MatrixXd A_null = lu.kernel();

    // Matrix representation of the conic
    B << A_null(0), A_null(1) / 2, A_null(3) / 2, A_null(1) / 2, A_null(2), A_null(4) / 2, A_null(3) / 2, A_null(4) / 2,
        A_null(5);

    return B;
}

Eigen::Matrix3d calculateG(Eigen::Matrix3d K1, Eigen::Matrix3d K2, Eigen::Matrix3d B) {
    // eq.17
    Eigen::Matrix3d D = DIAC(K1), Dp = DIAC(K2), I = Eigen::Matrix3d::Identity();
    // eq.18
    double t = (D * B).trace(), tp = (Dp * B).trace();
    // eq.19
    Eigen::Matrix3d U = 2 * D * B - t * I, Up = 2 * Dp * B - tp * I, G, Dpadj = Dp.adjoint(), Badj = B.adjoint(),
                    Dadj = D.adjoint();
    // eq.32
    G = 4 * U.transpose() * Dpadj * Badj * Dpadj * U + 2 * tp * U.transpose() * Dpadj * Up * U +
        4 * pow(tp, 2) * B * D * Dpadj * (D * B - t * I) - 4 * pow(tp, 2) * (Badj * Dpadj).trace() * Dadj +
        pow(tp, 4) * Dadj + pow(t, 2) * pow(tp, 2) * Dpadj;

    return G;
}

int _4pt_to_Ematrix(Eigen::Vector3d *P, Eigen::Vector3d *Q, Eigen::Matrix3d R, std::vector<Eigen::Matrix3d> *Es) {
    /*
				**Input: 4x Rotated points of view#1, 4x points in view#2, and the rotation matrix R
				**First we construct the 3x eq.46 vectors, one for each PC. Then computes the nullspace of this 4x6 matrix. Using this nullspace we compute *y1:6, z1:6, and w1:6 number, which are used to generate the conics (eq.49 and eq.50). Then, out of these conics we generate the quartic poly to find *the intersections, and the compute all the E1:6 numbers. Following eq.44, we construct the essential matrices, which are then multiplied from the right *side by the rotation matrix R.
     */
    Eigen::MatrixXd X_tilde(4, 6);
    Es->reserve(4);
    X_tilde << Q[0](0) * P[0](0), Q[0](0) * P[0](1), Q[0](1) * P[0](0), Q[0](1) * P[0](1), Q[0](2) * P[0](0),
        Q[0](2) * P[0](1), Q[1](0) * P[1](0), Q[1](0) * P[1](1), Q[1](1) * P[1](0), Q[1](1) * P[1](1),
        Q[1](2) * P[1](0), Q[1](2) * P[1](1), Q[2](0) * P[2](0), Q[2](0) * P[2](1), Q[2](1) * P[2](0),
        Q[2](1) * P[2](1), Q[2](2) * P[2](0), Q[2](2) * P[2](1), Q[3](0) * P[3](0), Q[3](0) * P[3](1),
        Q[3](1) * P[3](0), Q[3](1) * P[3](1), Q[3](2) * P[3](0), Q[3](2) * P[3](1);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd;
    svd.compute(X_tilde, Eigen::ComputeFullV);
    Eigen::MatrixXd V;
    V = svd.matrixV();

    double y1, y2, y3, y4, y5, y6, z1, z2, z3, z4, z5, z6, w1, w2, w3, w4, w5, w6;
    y1 = V.col(3)(0);
    y2 = V.col(3)(1);
    y3 = V.col(3)(2);
    y4 = V.col(3)(3);
    y5 = V.col(3)(4);
    y6 = V.col(3)(5);
    z1 = V.col(4)(0);
    z2 = V.col(4)(1);
    z3 = V.col(4)(2);
    z4 = V.col(4)(3);
    z5 = V.col(4)(4);
    z6 = V.col(4)(5);
    w1 = V.col(5)(0);
    w2 = V.col(5)(1);
    w3 = V.col(5)(2);
    w4 = V.col(5)(3);
    w5 = V.col(5)(4);
    w6 = V.col(5)(5);

    Eigen::Matrix3d C1, C2;
    C1 = essential_matrix_conic_1(y1, y2, y3, y4, y5, y6, z1, z2, z3, z4, z5, z6, w1, w2, w3, w4, w5, w6);
    C2 = essential_matrix_conic_2(y1, y2, y3, y4, y5, y6, z1, z2, z3, z4, z5, z6, w1, w2, w3, w4, w5, w6);

    Eigen::VectorXd QPC(5);

    QPC = Quartic_Poly_Coeff(C1(0, 0), 2 * C1(0, 1), C1(1, 1), 2 * C1(0, 2), 2 * C1(1, 2), C1(2, 2), C2(0, 0),
                             2 * C2(0, 1), C2(1, 1), 2 * C2(0, 2), 2 * C2(1, 2), C2(2, 2));
    QPC = Simplify_Poly_Coeff(QPC);
    // print_quartic_poly(QPC);
    Eigen::MatrixXd Cp(4, 4);
    Cp = Companion_Matrix(QPC);

    Eigen::EigenSolver<Eigen::MatrixXd> es(Cp);

    int num_sols = 0;

    for (int i = 0; i < es.eigenvalues().size(); i++) {
        double E1, E2, E3, E4, E5, E6;
        // check only real solutions
        if (es.eigenvalues()(i).imag() == 0.0) {


            // calculate the y coordinate by backsubstituting the z
            double y, z = es.eigenvalues()(i).real();
            y = y_coord_from_z(C1(0, 0), 2 * C1(0, 1), C1(1, 1), 2 * C1(0, 2), 2 * C1(1, 2), C1(2, 2), C2(0, 0),
                               2 * C2(0, 1), C2(1, 1), 2 * C2(0, 2), 2 * C2(1, 2), C2(2, 2), z);

            E1 = y1 * y + z1 * z + w1;
            E2 = y2 * y + z2 * z + w2;
            E3 = y3 * y + z3 * z + w3;
            E4 = y4 * y + z4 * z + w4;
            E5 = y5 * y + z5 * z + w5;
            E6 = y6 * y + z6 * z + w6;

            Eigen::Matrix3d E;
            E << E1, E2, 0, E3, E4, 0, E5, E6, 0;
            Es->emplace_back(E * R);
            num_sols++;
        }
    }
    return num_sols;
}

int solver_4pt_e(Eigen::Vector3d *p, Eigen::Vector3d *q, Eigen::Vector3d e, std::vector<Eigen::Matrix3d> *Es) {
    /*
				**Input: 4PC (p, q) + coordinates of the epipole of 1st view
				**Rotates the coordinate system of the first view, so that the epipole moves to (0, 0)
				**Then calls the _4pt_to_Ematrix function with input: the rotated points of view#1, the points of view#2, and the rotation matrix *Output: Up to four solutions for the Essential Matrix. It always returns 4 x Eigen::Matrix3d, and the ones that correspond to complex solutions are *zero matrices. Can be filtered using if(2 * Es[j] == Es[j]){continue;}.
     */
    Eigen::Vector3d en, PN[4];
    en = e / e.norm();

    // normalize the coordinates of all points
    for (int j = 0; j < 4; j++) {
        PN[j] = p[j] / p[j].norm();
    }

    // rotate points of the 1st view, os that the epipole lies on (0, 0)
    Eigen::Vector3d p0 = en;
    Eigen::Vector3d p1 = p0.cross(PN[0]);
    p1 = p1 / p1.norm();
    Eigen::Vector3d p2 = p0.cross(p1);
    p2 = p2 / p2.norm();
    Eigen::Matrix3d ZP, ZZ;
    ZP << p0, p1, p2;
    ZZ << 0, -1, 0, 0, 0, -1, 1, 0, 0;
    Eigen::Matrix3d CP = ZZ * ZP.transpose();

    // no need to rotate epipole (it will not be used)
    en = CP * en;
    en = en / en(2);

    // send point back to camera coordinates
    for (int j = 0; j < 4; j++) {
        PN[j] = CP * PN[j];
        PN[j] = PN[j] / PN[j](2);
    }
    int num_sols = _4pt_to_Ematrix(PN, q, CP, Es);

    return num_sols;
}
} // namespace threeview_nister

int threeview_nister(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                     const std::vector<Eigen::Vector2d> &x3, const Eigen::Vector3d &epipole,
                     const std::vector<size_t> &sample, bool use_enm, double sq_epipolar_error,
                     std::vector<ThreeViewCameraPose> *models) {
    Eigen::Vector3d p[4], q[4];
    std::vector<Eigen::Vector3d> x1n(4), x2n(4);

    models->resize(models->size() + 16);

    for (int k = 0; k < 4; ++k) {
        p[k] = x1[sample[k]].homogeneous();
        q[k] = x2[sample[k]].homogeneous();
        x1n[k] = p[k].normalized();
        x2n[k] = q[k].normalized();
    }

    std::vector<Eigen::Vector3d> x1s(3), x2s(3), x3s(3);

    for (int k = 0; k < 3; ++k) {
        x1s[k] = p[k];
        x2s[k] = q[k];
        x3s[k] = x3[sample[k]].homogeneous().normalized();
    }

    std::vector<Eigen::Matrix3d> Es;
    int num_sols = nister::solver_4pt_e(p, q, epipole, &Es);
    for (int j = 0; j < num_sols; j++) {
//        std::vector<char> l_inliers;
//        int l_num_inliers = get_inliers(Es[j], x1, x2, sq_epipolar_error, &l_inliers);
//        std::cout << "E12: " << std::endl << Es[j].normalized() << std::endl;
//        std::cout << "Sq threshold: " << sq_epipolar_error << std::endl;
//        std::cout << "Num inliers: " << l_num_inliers << std::endl;

        if ((2 * Es[j] - Es[j]).norm() < 1e-10)
            continue;

        const Eigen::Matrix3d E12 = Es[j];
        std::vector<CameraPose> poses12;

        if (use_enm) {
            poses12.resize(4);
            std::vector<char> inliers;
            int num_inliers = get_inliers(E12, x1, x2, sq_epipolar_error, &inliers);

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
                    poses12.emplace_back(pose);
                }
            }
        } else {
            motion_from_essential(E12, x1n, x2n, &poses12);
        }

        std::vector<Point3D> triangulated_12(3);

        for (const CameraPose& pose12 : poses12){
            for (size_t i = 0; i < 3; i++){
                triangulated_12[i] = triangulate(pose12, x1s[i], x2s[i]);
            }

            std::vector<CameraPose> models13;
            p3p(x3s, triangulated_12, &models13);

            for (const CameraPose& pose13 : models13){
                ThreeViewCameraPose three_view_pose = ThreeViewCameraPose(pose12, pose13);
                models->emplace_back(three_view_pose);
            }
        }
    }
    return models->size();
}

} // namespace poselib

