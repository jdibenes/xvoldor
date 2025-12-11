//
// Created by kocur on 29-Oct-24.
//

#include "threeview_para.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/utils.h"

#include <iostream>

namespace poselib {

void weight_points(const Eigen::Matrix<double, 1, 3> & z,
                   const Eigen::Matrix<double, 2, 4> & x, Eigen::Matrix<double, 2, 4> * xp)
{
    xp->col(0) = x.col(0);
    xp->col(1) = x.col(1) * (1 + z(0));
    xp->col(2) = x.col(2) * (1 + z(1));
    xp->col(3) = x.col(3) * (1 + z(2));
}

int solver_4p3v(const Eigen::Matrix<double, 2, 4> & x1,
                const Eigen::Matrix<double, 2, 4> & x2,
                const Eigen::Matrix<double, 2, 4> & x3,
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
        } else {
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

        solve_for_translation(x1c,x2c,x3c,s.R2,s.R3,&s.t2,&s.t3);
    }


    return 0;
}

double solver_4p3v_para(const Eigen::Matrix<double, 2, 4> &x1,
                        const Eigen::Matrix<double, 2, 4> &x2,
                        const Eigen::Matrix<double, 2, 4> &x3,
                        Solution *sol)
{

    Eigen::Matrix<double, 2, 3> A, B, C;
    A << x1.col(1)-x1.col(0), x1.col(2)-x1.col(0), x1.col(3)-x1.col(0);
    B << x2.col(1)-x2.col(0), x2.col(2)-x2.col(0), x2.col(3)-x2.col(0);
    C << x3.col(1)-x3.col(0), x3.col(2)-x3.col(0), x3.col(3)-x3.col(0);

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

    double rv = v(1)/v(0);
    double ru = u(1)/u(0);
    double ruv = u(0)/v(0);
    double rvu2 = v(1) / u(1);
    double rvp = vp(1)/vp(0);
    double rup = up(1)/up(0);
    double ruvp = up(0)/vp(0);
    double rvup2 = vp(1) / up(1);
    double rvpp = vpp(1)/vpp(0);
    double rupp = upp(1)/upp(0);
    double ruvpp = upp(0)/vpp(0);
    double rvupp2 = vpp(1) / upp(1);



    double T12 = std::sqrt((rv*rv+1)/((ru*ru+1)*(ruv*ruv)));
    double T31 = std::sqrt((rvp*rvp+1)/((rup*rup+1)*ruvp*ruvp));
    double T23 = std::sqrt((rvpp*rvpp+1)/((rupp*rupp+1)*ruvpp*ruvpp));

    Eigen::Matrix<double, 4, 1> nr, ns, nv;
    nr << 1.0, ru, (1.0/T12)*ru*rvu2/rv, (1.0/T12)*ru*rvu2;
    ns << 1.0, rup, (1.0/T31)*rup*rvup2/rvp, (1.0/T31)*rup*rvup2;
    nv << 1.0, rupp, (1.0/T23)*rupp*rvupp2/rvpp, (1.0/T23)*rupp*rvupp2;

    double pq1 = (nr(2)+nr(0))/(nr(3)+nr(1));
    double rs1 = (nr(2)+nr(0))/(nr(3)-nr(1));
    double pq2 = (ns(2)+ns(0))/(ns(3)+ns(1));
    double rs2 = (ns(2)+ns(0))/(ns(3)-ns(1));
    double pq3 = (nv(2)+nv(0))/(nv(3)+nv(1));
    double rs3 = (nv(2)+nv(0))/(nv(3)-nv(1));

    double A1 = std::atan(pq1);
    double B1 = std::atan(rs1);
    double A2 = std::atan(pq2);
    double B2 = std::atan(rs2);
    double A3 = std::atan(pq3);
    double B3 = std::atan(rs3);

    double S0 = std::sin(B1+B2+B3);
    double S1 = std::sin(B1-A2+A3);
    double S2 = std::sin(B2-A3+A1);
    double S3 = std::sin(B3-A1+A2);

    double tC1 = std::sqrt(std::abs((S0*S1)/(S2*S3)));


    //for(int i = 0; i < 2; ++i) {

    double C1 = std::atan(tC1);
    double C2 = std::atan(tC1*(S2/S1));
    //double C3 = std::atan(tC1*(S3/S1));

    Eigen::Quaternion<double> qr, qs, qv;
    qr.coeffs() << std::sin(A1)*std::sin(C1),
        std::cos(A1)*std::sin(C1),
        std::sin(B1)*std::cos(C1),
        std::cos(B1)*std::cos(C1);



    qs.coeffs() << std::sin(A2)*std::sin(C2),
        std::cos(A2)*std::sin(C2),
        std::sin(B2)*std::cos(C2),
        -std::cos(B2)*std::cos(C2);

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

    return std::abs(1.0 - T12*T23*T31);
}


void centering_rotation(const Eigen::Vector3d &x0, Eigen::Matrix3d* R)
{

    Eigen::Vector3d r3 = x0.normalized();
    Eigen::Vector3d r1{ -r3(2),0.0,r3(0) };
    r1.normalize();
    Eigen::Vector3d r2 = r3.cross(r1);

    R->row(0) = r1;
    R->row(1) = r2;
    R->row(2) = r3;
}



void solve_for_translation(const Eigen::Matrix<double, 2, 4> &x1, const Eigen::Matrix<double, 2, 4> &x2,
                           const Eigen::Matrix<double, 2, 4> &x3, const Eigen::Matrix3d &R2, const Eigen::Matrix3d &R3,
                           Eigen::Matrix<double, 3, 1> *t2, Eigen::Matrix<double, 3, 1> *t3)
{

    Eigen::Matrix<double, 3, 4> X1 = x1.colwise().homogeneous();
    Eigen::Matrix<double, 3, 4> X2 = x2.colwise().homogeneous();
    Eigen::Matrix<double, 3, 4> X3 = x3.colwise().homogeneous();

    Eigen::Matrix<double, 12, 6> A;
    A.setZero();

    for(int i = 0; i < 4; ++i) {
        A.block<1, 3>(i, 0) = (R2 * X1.col(i)).cross(X2.col(i));
        A.block<1, 3>(4+i, 3) = (R3 * X1.col(i)).cross(X3.col(i));

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
    Eigen::Vector3d X = R2*x1.col(0).homogeneous();
    double lambda = (x2.col(0).dot(t2->block<2,1>(0,0)) - (*t2)(2) * n2 ) / ( n2 * X(2) - x2.col(0).dot(X.block<2,1>(0,0)) );
    if(lambda < 0) {
        *t2 *= -1.0;
        *t3 *= -1.0;
    }

}

void solver_4p3v_para(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                      const std::vector<Eigen::Vector2d> &x3, const std::vector<size_t> &sample,
                      std::vector<ThreeViewCameraPose> *models, int iters, double sq_epipolar_t){
    Eigen::Matrix<double, 2, 4> xx1, xx2, xx3;

    for (int k = 0; k < 4; ++k){
        xx1(0, k) = x1[sample[k]][0];
        xx1(1, k) = x1[sample[k]][1];

        xx2(0, k) = x2[sample[k]][0];
        xx2(1, k) = x2[sample[k]][1];

        xx3(0, k) = x3[sample[k]][0];
        xx3(1, k) = x3[sample[k]][1];
    }

    ThreeViewsVector solutions;

    solver_4p3v(xx1, xx2, xx3, &solutions, iters);

    models->reserve(solutions.size());

    for (ThreeViews sol : solutions){
        Eigen::Matrix3d E12, E13;
        essential_from_motion(CameraPose(sol.R2, sol.t2), &E12);
        essential_from_motion(CameraPose(sol.R3, sol.t3), &E13);

        std::cout << "R2: " << std::endl << sol.R2 << std::endl;
        std::cout << "t2: " << sol.t2.transpose() << std::endl;
        std::cout << "R3:" << std::endl << sol.R3 << std::endl;
        std::cout << "t3: " << sol.t3.transpose() << std::endl;

        std::vector<char> inliers_12, inliers_13;
        int num_inliers_12 = get_inliers(E12, x1, x2, sq_epipolar_t, &inliers_12);
        int num_inliers_13 = get_inliers(E13, x1, x3, sq_epipolar_t, &inliers_13);

        std::cout << "Num inliers 12: " << num_inliers_12 << " Num inliers 13: " << num_inliers_13 << std::endl;

        models->emplace_back(ThreeViewCameraPose(CameraPose(sol.R2, sol.t2), CameraPose(sol.R3, sol.t3)));
    }
}
} // namespace poselib