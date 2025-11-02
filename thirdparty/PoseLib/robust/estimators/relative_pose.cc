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

#include "relative_pose.h"

#include "PoseLib/misc/decompositions.h"
#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/gen_relpose_5p1pt.h"
#include "PoseLib/solvers/homography_4pt.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/relpose_5pt.h"
#include "PoseLib/solvers/relpose_6pt_focal.h"
#include "PoseLib/solvers/relpose_7pt.h"
#include "PoseLib/solvers/relpose_affine_4p.h"
#include "PoseLib/solvers/threeview_nister.h"
#include "PoseLib/solvers/threeview_para.h"

#include <iostream>
#include <random>
#include <torch/script.h>

namespace poselib {

Point2D RelativePoseEstimator::triangle_calc() {
    double mx2 = m2(0);
    double my2 = m2(1);
    double tr_xA = x2[sample[0]](0), tr_yA = x2[sample[0]](1);
    double tr_xB = x2[sample[1]](0), tr_yB = x2[sample[1]](1);
    double tr_xC = x2[sample[2]](0), tr_yC = x2[sample[2]](1);

    double slopeAB = (tr_yB - tr_yA) / (tr_xB - tr_xA);
    double slopeAC = (tr_yC - tr_yA) / (tr_xC - tr_xA);
    double slopeBC = (tr_yC - tr_yB) / (tr_xC - tr_xB);

    double interceptAB = tr_yA - slopeAB * tr_xA;
    double interceptAC = tr_yA - slopeAC * tr_xA;
    double interceptBC = tr_yB - slopeBC * tr_xB;

    // find y coord in lines AB, AC, and BC, in the 2nd view for x=mean
    double yAB = slopeAB * mx2 + interceptAB;
    double yAC = slopeAC * mx2 + interceptAC;
    double yBC = slopeBC * mx2 + interceptBC;

    // find x coord in lines AB, AC, and BC, in the 2nd view for y=mean
    double xAB = (my2 - interceptAB) / slopeAB;
    double xAC = (my2 - interceptAC) / slopeAC;
    double xBC = (my2 - interceptBC) / slopeBC;

    // find min and max y and min and max x of thriangle vertices
    double min_y = std::min(std::min(tr_yA, tr_yB), tr_yC);
    double max_y = std::max(std::max(tr_yA, tr_yB), tr_yC);
    double min_x = std::min(std::min(tr_xA, tr_xB), tr_xC);
    double max_x = std::max(std::max(tr_xA, tr_xB), tr_xC);

    double max_distY;
    if (yAB>max_y || yAB<min_y) {
        max_distY = std::abs(yAC-yBC);
    } else if (yAC>max_y || yAC<min_y) {
        max_distY = std::abs(yAB-yBC);
    } else {
        max_distY = std::abs(yAB-yAC);
    }

    double max_distX;
    if (xAB>max_x || xAB<min_x) {
        max_distX = std::abs(xAC-xBC);
    } else if (xAC>max_x || xAC<min_x) {
        max_distX = std::abs(xAB-xBC);
    } else {
        max_distX = std::abs(xAB-xAC);
    }
    if (max_distX > max_distY){
        return Point2D(max_distX, 0);
    } else {
        return Point2D(0, max_distY);
    }
}

void RelativePoseEstimator::non_minimal_refinement(std::vector<CameraPose> *models) const{
    CameraPoseVector new_models;
    new_models.reserve(models->size());
    for (CameraPose model : *models) {
        std::vector<char> inliers;
        int num_inliers = get_inliers(model, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, &inliers);

        if (num_inliers > 5) {
            std::vector<Eigen::Vector3d> x1_inlier, x2_inlier;
            x1_inlier.reserve(num_inliers);
            x2_inlier.reserve(num_inliers);
            for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
                if (inliers[pt_k]) {
                    x1_inlier.emplace_back(x1[pt_k].homogeneous().normalized());
                    x2_inlier.emplace_back(x2[pt_k].homogeneous().normalized());
                }
            }
            relpose_5pt(x1_inlier, x2_inlier, &new_models);
        }
    }
    *models = new_models;
}

void RelativePoseEstimator::lm_refinement(std::vector<CameraPose> *models) const{
    CameraPoseVector new_models;
    new_models.reserve(models->size());
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    // Find approximate inliers and bundle over these with a truncated loss
    for (CameraPose &pose: *models) {
        std::vector<char> inliers;
        int num_inl = get_inliers(pose, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
        std::vector<Eigen::Vector2d> x1_inlier, x2_inlier;
        x1_inlier.reserve(num_inl);
        x2_inlier.reserve(num_inl);

        if (num_inl > 5) {
            for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
                if (inliers[pt_k]) {
                    x1_inlier.push_back(x1[pt_k]);
                    x2_inlier.push_back(x2[pt_k]);
                }
            }
            refine_relpose(x1_inlier, x2_inlier, &pose, bundle_opt);
            new_models.emplace_back(pose);
        }
    }
    *models = new_models;
}

void RelativePoseEstimator::generate_models(std::vector<CameraPose> *models){
    sampler.generate_sample(&sample);

    if (opt.use_affine) {
        switch(sample_sz){
        case 2:
            affine_essential_2p(x1, x2, sample, opt.max_epipolar_error * opt.max_epipolar_error, opt.early_nonminimal, models);
            break;
        case 3:
            affine_homography_3p(x1, x2, sample, opt.max_epipolar_error * opt.max_epipolar_error, opt.early_nonminimal, models);
            break;
        case 4:
            relpose_affine_4p(x1, x2, sample, opt.max_epipolar_error * opt.max_epipolar_error, opt.early_nonminimal, models);
            break;
        default:
            throw std::runtime_error("Twoview affine invoked with wrong sample sz");
        }
        return;
    }

    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }

    //
    if (opt.use_homography){
        // use virtual correspondence
        if (sample_sz == 3) {
            m1 = (x1[sample[0]] + x1[sample[1]] + x1[sample[2]])/3;
            m2 = (x2[sample[0]] + x2[sample[1]] + x2[sample[2]])/3;
            x1s[3] = m1.homogeneous().normalized();
            x2s[3] = m2.homogeneous().normalized();
        }

        Eigen::Matrix3d H;
        int sols = homography_4pt(x1s, x2s, &H, true);
        if (sols > 0)
            pose_from_H(H, models);

        if (sample_sz == 3 and opt.delta > 0.0) {
            Point2D dir = triangle_calc();
            x2s[3] = (m2 + opt.delta * dir).homogeneous().normalized();
            sols = homography_4pt(x1s, x2s, &H, true);
            if (sols > 0)
                pose_from_H(H, models);

            x2s[3] = (m2 - opt.delta * dir).homogeneous().normalized();
            sols = homography_4pt(x1s, x2s, &H, true);
            if (sols > 0)
                pose_from_H(H, models);
        }

        if (opt.early_nonminimal){
            non_minimal_refinement(models);
        }
        if (opt.early_lm){
            lm_refinement(models);
        }
        return;
    }

    if (sample_sz == 4){
        m1 = (x1[sample[0]] + x1[sample[1]] + x1[sample[2]])/3;
        m2 = (x2[sample[0]] + x2[sample[1]] + x2[sample[2]])/3;
        x1s[4] = m1.homogeneous().normalized();
        x2s[4] = m2.homogeneous().normalized();
    }

    relpose_5pt(x1s, x2s, models);

//    if (opt.all_triangles){
//        Point2D all1 = x1[sample[0]] + x1[sample[1]] + x1[sample[3]] + x1[sample[4]];
//        Point2D all2 = x2[sample[0]] + x2[sample[1]] + x2[sample[3]] + x2[sample[4]];
//
//        std::vector<CameraPose> all_models;
//
//        for (int i = 0; i < 4; ++i) {
//            x1s[4] = ((all1 - x1[sample[i]]) / 3).homogeneous().normalized();
//            x2s[4] = ((all2 - x2[sample[i]]) / 3).homogeneous().normalized();
//            relpose_5pt(x1s, x2s, &all_models);
//            models->reserve(models->size() + all_models.size());
//            models->insert(models->end(), all_models.begin(), all_models.end());
//        }
//    }

    if (opt.delta > 0.0){
        Point2D dir = triangle_calc();

        std::vector<CameraPose> delta_models;

        x2s[4] = (m2 + opt.delta * dir).homogeneous().normalized();
        relpose_5pt(x1s, x2s, &delta_models);
        models->reserve(models->size() + delta_models.size());
        models->insert(models->end(), delta_models.begin(), delta_models.end());

        x2s[4] = (m2 - opt.delta * dir).homogeneous().normalized();
        relpose_5pt(x1s, x2s, &delta_models);
        models->reserve(models->size() + delta_models.size());
        models->insert(models->end(), delta_models.begin(), delta_models.end());
    }

    if (opt.early_nonminimal){
        non_minimal_refinement(models);
    }

    if (opt.early_lm){
        lm_refinement(models);
    }
}
void RelativePoseEstimator::pose_from_H(Eigen::Matrix3d &H, std::vector<CameraPose> *models) const {
    std::vector<CameraPose> poses;
    std::vector<Eigen::Vector3d> ns;
    motion_from_homography_svd(H, poses, ns);
    models->reserve(models->size() + poses.size());
    models->insert(models->end(), poses.begin(), poses.end());
}

double RelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_sampson_msac_score(pose, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void RelativePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*pose, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);

    if (num_inl <= 5) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
        }
    }

    refine_relpose(x1_inlier, x2_inlier, pose, bundle_opt);
}

Eigen::Vector3d ThreeViewRelativePoseEstimator::get_network_point(){
    std::vector<Eigen::Vector3f> P(5);
    std::vector<Eigen::Vector3f> Q(5);
    std::vector<Eigen::Vector3f> R(5);

    for (size_t k = 0; k < 4; ++k){
        P[k] = x1[sample[k]].homogeneous().cast<float>();
        Q[k] = x2[sample[k]].homogeneous().cast<float>();
        R[k] = x3[sample[k]].homogeneous().cast<float>();
    }
    P[4] = ((x1[sample[0]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().cast<float>();
    Q[4] = ((x2[sample[0]] + x2[sample[1]] + x2[sample[2]]) / 3.0).homogeneous().cast<float>();
    R[4] = ((x3[sample[0]] + x3[sample[1]] + x3[sample[2]]) / 3.0).homogeneous().cast<float>();

    std::vector<Eigen::Vector2f>  P1(5), Q1(5), R1(5);
    Eigen::Matrix3f RP1, RQ1, RR1;

    normalize(P, Q, R, P1, Q1, R1, RP1, RQ1, RR1);

//    std::cout << "RR1" << std::endl << RR1 << std::endl;

    float data[] = {P1[0][0], P1[0][1], Q1[0][0], Q1[0][1], R1[0][0], R1[0][1],
                    P1[1][0], P1[1][1], Q1[1][0], Q1[1][1], R1[1][0], R1[1][1],
                    P1[2][0], P1[2][1], Q1[2][0], Q1[2][1], R1[2][0], R1[2][1],
                    P1[3][0], P1[3][1], Q1[3][0], Q1[3][1], R1[3][0], R1[3][1]};

    std::vector<torch::jit::IValue> inputs;
    torch::Tensor f = torch::from_blob(data, {1, 4, 6});
    inputs.push_back(f);

    at::Tensor output_tensor;
    output_tensor = module.forward(inputs).toTensor();
    // vectorize prediction
    Eigen::Vector3f pred1, pred2;

    pred1 << output_tensor[0][0].item<float>(), output_tensor[0][1].item<float>(), 1.0;
    pred1 = pred1 / pred1.norm();
    Eigen::Vector3f rp = RQ1.transpose() * pred1;
    return rp.normalized().cast<double>();
}

void ThreeViewRelativePoseEstimator::generate_nn_init_delta_models(std::vector<ThreeViewCameraPose> *models){
    std::vector<Eigen::Vector3f> P(5);
    std::vector<Eigen::Vector3f> Q(5);
    std::vector<Eigen::Vector3f> R(5);

    for (size_t k = 0; k < 4; ++k){
        P[k] = x1[sample[k]].homogeneous().cast<float>();
        Q[k] = x2[sample[k]].homogeneous().cast<float>();
        R[k] = x3[sample[k]].homogeneous().cast<float>();
    }
    P[4] = ((x1[sample[0]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().cast<float>();
    R[4] = ((x3[sample[0]] + x3[sample[1]] + x3[sample[2]]) / 3.0).homogeneous().cast<float>();

    std::vector<Eigen::Vector2f>  P1(5), Q1(5), R1(5);
    Eigen::Matrix3f RP1, RQ1, RR1;

    Eigen::Vector2d q4 = (x2[sample[0]] + x2[sample[1]] + x2[sample[2]]) / 3.0;
    int idx;
    double scale;
    triangle_calc(q4(0), q4(1), idx, scale);

    for (size_t k = 0; k < 3; ++k){
        if (k == 0)
            Q[4] = q4.homogeneous().cast<float>();
        if (k == 1) {
            q4[idx] += scale * opt.delta;
            Q[4] = q4.homogeneous().cast<float>();
        }
        if (k == 2){
            q4[idx] -= 2 * scale * opt.delta;
            Q[4] = q4.homogeneous().cast<float>();
        }

        normalize(P, Q, R, P1, Q1, R1, RP1, RQ1, RR1);

        float data[] = {P1[0][0], P1[0][1], Q1[0][0], Q1[0][1], R1[0][0], R1[0][1],
                        P1[1][0], P1[1][1], Q1[1][0], Q1[1][1], R1[1][0], R1[1][1],
                        P1[2][0], P1[2][1], Q1[2][0], Q1[2][1], R1[2][0], R1[2][1],
                        P1[3][0], P1[3][1], Q1[3][0], Q1[3][1], R1[3][0], R1[3][1]};

        std::vector<torch::jit::IValue> inputs;
        torch::Tensor f = torch::from_blob(data, {1, 4, 6});
        inputs.emplace_back(f);

        at::Tensor output_tensor;
        output_tensor = module.forward(inputs).toTensor();
        Eigen::Vector3f pred1, pred2;
        pred1 << output_tensor[0][0].item<float>(), output_tensor[0][1].item<float>(), 1.0;
        pred1 = pred1 / pred1.norm();
        Eigen::Vector3f rp = RQ1.transpose() * pred1;
        x2n[4] = rp.normalized().cast<double>();
//        std::cout << "Predicted: " << x2n[4](0)/x2n[4](2) << ", " << x2n[4](1)/x2n[4](2) << std::endl;
//        x2n[4] = q4.homogeneous().normalized().cast<double>();
        estimate_models(models);
    }
}

void ThreeViewRelativePoseEstimator::non_minimal_refinement(std::vector<CameraPose> *models) const{
    CameraPoseVector new_models;
    new_models.reserve(models->size());
    for (CameraPose model : *models) {
        std::vector<char> inliers;
        int num_inliers = get_inliers(model, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, &inliers);

        if (num_inliers > 5) {
            std::vector<Eigen::Vector3d> x1_inlier, x2_inlier;
            x1_inlier.reserve(num_inliers);
            x2_inlier.reserve(num_inliers);
            for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
                if (inliers[pt_k]) {
                    x1_inlier.emplace_back(x1[pt_k].homogeneous().normalized());
                    x2_inlier.emplace_back(x2[pt_k].homogeneous().normalized());
                }
            }
            relpose_5pt(x1_inlier, x2_inlier, &new_models);
        }
    }
    *models = new_models;
}

void ThreeViewRelativePoseEstimator::lm_refinement(std::vector<CameraPose> *models) const{
    CameraPoseVector new_models;
    new_models.reserve(models->size());
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    // Find approximate inliers and bundle over these with a truncated loss
    for (CameraPose &pose: *models) {
        std::vector<char> inliers;
        int num_inl = get_inliers(pose, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
        std::vector<Eigen::Vector2d> x1_inlier, x2_inlier;
        x1_inlier.reserve(num_inl);
        x2_inlier.reserve(num_inl);

        if (num_inl > 5) {
            for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
                if (inliers[pt_k]) {
                    x1_inlier.push_back(x1[pt_k]);
                    x2_inlier.push_back(x2[pt_k]);
                }
            }
            refine_relpose(x1_inlier, x2_inlier, &pose, bundle_opt);
            new_models.emplace_back(pose);
        }
    }
    *models = new_models;
}

void ThreeViewRelativePoseEstimator::generate_models(std::vector<ThreeViewCameraPose> *models) {
    sampler.generate_sample(&sample);

    if (opt.use_para) {
        solver_4p3v_para(x1, x2, x3, sample, models, 0, opt.max_epipolar_error * opt.max_epipolar_error);
        return;
    }

    if (opt.use_nister > 0) {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, opt.nister_scale);

        double angle = std::rand() * 2 * 3.14159265358979323846;
        Eigen::Vector3d noisy_epipole = epipole;
        std::cout << "Nister scale: " << opt.nister_scale << std::endl;
        noisy_epipole[0] += 200 * std::cos(angle) * opt.nister_scale;
        noisy_epipole[1] += 200 * std::sin(angle) * opt.nister_scale;
//        noisy_epipole[0] = 0;
//        noisy_epipole[1] = 0;

        threeview_nister(x1, x2, x3, noisy_epipole, sample, opt.early_nonminimal,
                             opt.max_epipolar_error * opt.max_epipolar_error, models);
        if (opt.use_nister == 1) {
            return;
        }

        angle = std::rand() * 2 * 3.14159265358979323846;
        noisy_epipole = epipole;
        noisy_epipole[0] += 40 * std::cos(angle) * opt.nister_scale;
        noisy_epipole[1] += 40 * std::sin(angle) * opt.nister_scale;

        threeview_nister(x1, x2, x3, noisy_epipole, sample, opt.early_nonminimal,
                         opt.max_epipolar_error * opt.max_epipolar_error, models);

        angle = std::rand() * 2 * 3.14159265358979323846;
        noisy_epipole = epipole;
        noisy_epipole[0] += 20 * std::cos(angle) * opt.nister_scale;
        noisy_epipole[1] += 20 * std::sin(angle) * opt.nister_scale;

        threeview_nister(x1, x2, x3, noisy_epipole, sample, opt.early_nonminimal,
                         opt.max_epipolar_error * opt.max_epipolar_error, models);
        return;
    }

    for (size_t k = 0; k < sample_sz_13; ++k) {
        x1s[k] = x1[sample[k]].homogeneous();
        x2s[k] = x2[sample[k]].homogeneous();
        x3s[k] = x3[sample[k]].homogeneous().normalized();
    }

//    std::cout << "GT E: " << std::endl << opt.gt_E << std::endl;

    if (opt.use_affine){
        estimate_models(models);
        return;
    }

    for (size_t k = 0; k < sample_sz; ++k) {
        x1n[k] = x1[sample[k]].homogeneous().normalized();
        x2n[k] = x2[sample[k]].homogeneous().normalized();
    }

    if (sample_sz == 4){
        x1n[4] = ((x1[sample[0]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().normalized();
        if (opt.init_net){
            generate_nn_init_delta_models(models);
            return;
        }

        if (opt.oracle){
            Eigen::Vector3d epipolar_line = (opt.gt_E * x1n[4]).normalized();

            double x_0 = (x2[sample[0]](0) + x2[sample[1]](0) + x2[sample[2]](0)) / 3.0;
            double y = - (epipolar_line(2) + epipolar_line(0) * x_0) / epipolar_line(1);
            x2n[4] = Eigen::Vector2d(x_0, y).homogeneous().normalized();
            estimate_models(models);

            return;
        }

        if (opt.use_net){
            x2n[4] = get_network_point();
        } else {
            x2n[4] = ((x2[sample[0]] + x2[sample[1]] + x2[sample[2]]) / 3.0).homogeneous().normalized();
        }
    }

    estimate_models(models);

    if (sample_sz == 4 and opt.delta > 0.0){
        delta(x2n[4](0) / x2n[4](2), x2n[4](1) / x2n[4](2), models);
    }
}

int ThreeViewRelativePoseEstimator::normalize(std::vector<Eigen::Vector3f> &P, std::vector<Eigen::Vector3f> &Q,
                                              std::vector<Eigen::Vector3f> &T, std::vector<Eigen::Vector2f> &P1,
                                              std::vector<Eigen::Vector2f> &Q1, std::vector<Eigen::Vector2f> &T1,
                                              Eigen::Matrix3f &CP1, Eigen::Matrix3f &CQ1, Eigen::Matrix3f &CT1) const
{
//    Eigen::Vector3f centroidP = Eigen::Vector3f::Zero();
//    Eigen::Vector3f centroidQ = Eigen::Vector3f::Zero();
    for(int i=0;i<5;++i){
        P[i] = P[i]/P[i].norm();
        Q[i] = Q[i]/Q[i].norm();
        T[i] = T[i]/T[i].norm();
    }

    // generate the rotation matrix that sends first point to (0, 0) and second point to (0, x)

    // for view #1
    Eigen::Vector3f p0 = P[4]/P[4].norm();
    Eigen::Vector3f p1 = p0.cross(P[3]);
    p1 = p1/p1.norm();
    Eigen::Vector3f p2 = p0.cross(p1);
    p2 = p2/p2.norm();
    Eigen::Matrix3f Zp;
    Zp << p0, p1, p2;

    // for view #2
    Eigen::Vector3f q0 = Q[4]/Q[4].norm();
    Eigen::Vector3f q1 = q0.cross(Q[3]);
    q1 = q1/q1.norm();
    Eigen::Vector3f q2 = q0.cross(q1);
    q2 = q2/q2.norm();
    Eigen::Matrix3f Zq;
    Zq << q0, q1, q2;

    // for view #3
    Eigen::Vector3f t0 = T[4]/T[4].norm();
    Eigen::Vector3f t1 = t0.cross(T[3]);
    t1 = t1/t1.norm();
    Eigen::Vector3f t2 = t0.cross(t1);
    t2 = t2/t2.norm();
    Eigen::Matrix3f Zt;
    Zt << t0, t1, t2;

    Eigen::Matrix3f ZZ;
    ZZ << 0,-1,0,0,0,-1,1,0,0;
    Eigen::Matrix3f CP = ZZ * Zp.transpose();
    Eigen::Matrix3f CQ = ZZ * Zq.transpose();
    Eigen::Matrix3f CT = ZZ * Zt.transpose();

    // rotate the points and project them back to the plane
    for(int i=0;i<5;++i){
        P[i] = CP * P[i];
        P[i] = P[i]/P[i](2);

        Q[i] = CQ * Q[i];
        Q[i] = Q[i]/Q[i](2);

        T[i] = CT * T[i];
        T[i] = T[i]/T[i](2);
    }

    // hardcode the "0" values because of the limitation of numerical precision
    //P[0][0] = 0.0;
    //P[0][1] = 0.0;
    //P[1][0] = 0.0;
    P[3][0] = 0.0;

    //Q[0][0] = 0.0;
    //Q[0][1] = 0.0;
    //Q[1][0] = 0.0;
    Q[3][0] = 0.0;

    //T[0][0] = 0.0;
    //T[0][1] = 0.0;
    //T[1][0] = 0.0;
    T[3][0] = 0.0;

    for(int i=0;i<5;++i){
        Eigen::Vector2f p;
        p(0) = P[i](0);
        p(1) = P[i](1);
        P1[i] = p;

        Eigen::Vector2f q;
        q(0) = Q[i](0);
        q(1) = Q[i](1);
        Q1[i] = q;

        Eigen::Vector2f t;
        t(0) = T[i](0);
        t(1) = T[i](1);
        T1[i] = t;
    }
    CP1 = CP;
    CQ1 = CQ;
    CT1 = CT;

    return 1;
}

int ThreeViewRelativePoseEstimator::normalize(std::vector<Eigen::Vector3f> &P, std::vector<Eigen::Vector3f> &Q,
                                              std::vector<Eigen::Vector2f> &P1, std::vector<Eigen::Vector2f> &Q1,
                                              Eigen::Matrix3f &CP1,Eigen::Matrix3f &CQ1) const {
//    Eigen::Vector3f centroidP = Eigen::Vector3f::Zero();
//    Eigen::Vector3f centroidQ = Eigen::Vector3f::Zero();
    for(int i=0;i<5;++i){
        P[i] = P[i]/P[i].norm();
        Q[i] = Q[i]/Q[i].norm();
    }

    // generate the rotation matrix that sends first point to (0, 0) and second point to (0, x)

    // for view #1
    Eigen::Vector3f p0 = P[0]/P[0].norm();
    Eigen::Vector3f p1 = p0.cross(P[1]);
    p1 = p1/p1.norm();
    Eigen::Vector3f p2 = p0.cross(p1);
    p2 = p2/p2.norm();
    Eigen::Matrix3f Zp;
    Zp << p0, p1, p2;

    // for view #2
    Eigen::Vector3f q0 = Q[0]/Q[0].norm();
    Eigen::Vector3f q1 = q0.cross(Q[1]);
    q1 = q1/q1.norm();
    Eigen::Vector3f q2 = q0.cross(q1);
    q2 = q2/q2.norm();
    Eigen::Matrix3f Zq;
    Zq << q0, q1, q2;

    Eigen::Matrix3f ZZ;
    ZZ << 0,-1,0,0,0,-1,1,0,0;
    Eigen::Matrix3f CP = ZZ * Zp.transpose();
    Eigen::Matrix3f CQ = ZZ * Zq.transpose();


    // rotate the points and project them back to the plane
    for(int i=0;i<5;++i){
        P[i] = CP * P[i];
        P[i] = P[i]/P[i](2);

        Q[i] = CQ * Q[i];
        Q[i] = Q[i]/Q[i](2);
    }

    // hardcode the "0" values because of the limitation of numerical precision
    P[0][0] = 0.0;
    P[0][1] = 0.0;
    P[1][0] = 0.0;

    Q[0][0] = 0.0;
    Q[0][1] = 0.0;
    Q[1][0] = 0.0;

    for(int i=0;i<5;++i){
        Eigen::Vector2f p;
        p(0) = P[i](0);
        p(1) = P[i](1);
        P1[i] = p;

        Eigen::Vector2f q;
        q(0) = Q[i](0);
        q(1) = Q[i](1);
        Q1[i] = q;
    }
    CP1 = CP;
    CQ1 = CQ;

    return 1;
}

void ThreeViewRelativePoseEstimator::delta(double mx, double my, std::vector<ThreeViewCameraPose> *models) {
    int idx;
    double scale;
    triangle_calc(mx, my, idx, scale);

    //    Point2D x2n4 = ((x2[sample[0]] + x2[sample[1]] + x2[sample[2]]) / 3.0);
//
//    double min_x = std::min({x2[sample[0]][0], x2[sample[1]][0], x2[sample[2]][0]});
//    double min_y = std::min({x2[sample[0]][1], x2[sample[1]][1], x2[sample[2]][1]});
//    double max_x = std::max({x2[sample[0]][0], x2[sample[1]][0], x2[sample[2]][0]});
//    double max_y = std::max({x2[sample[0]][1], x2[sample[1]][1], x2[sample[2]][1]});
//
//    int idx;
//    double scale;
//    if ((max_x - min_x) > (max_y - min_y)) {
//        scale = max_x - min_x;
//        idx = 0;
//    } else {
//        scale = max_y - min_y;
//        idx = 1;
//    }

    Eigen::Vector2d x2n4(mx, my);
    x2n4(idx) += opt.delta * scale;
    x2n[4] = x2n4.homogeneous().normalized();
    estimate_models(models);

    x2n4(idx) -= 2 * opt.delta * scale;
    x2n[4] = x2n4.homogeneous().normalized();
    estimate_models(models);
}

void ThreeViewRelativePoseEstimator::triangle_calc(double mx2, double my2, int &idx, double &scale) {
    double tr_xA = x2[sample[0]](0), tr_yA = x2[sample[0]](1);
    double tr_xB = x2[sample[1]](0), tr_yB = x2[sample[1]](1);
    double tr_xC = x2[sample[2]](0), tr_yC = x2[sample[2]](1);

    double slopeAB = (tr_yB - tr_yA) / (tr_xB - tr_xA);
    double slopeAC = (tr_yC - tr_yA) / (tr_xC - tr_xA);
    double slopeBC = (tr_yC - tr_yB) / (tr_xC - tr_xB);

    double interceptAB = tr_yA - slopeAB * tr_xA;
    double interceptAC = tr_yA - slopeAC * tr_xA;
    double interceptBC = tr_yB - slopeBC * tr_xB;

    // find y coord in lines AB, AC, and BC, in the 2nd view for x=mean
    double yAB = slopeAB * mx2 + interceptAB;
    double yAC = slopeAC * mx2 + interceptAC;
    double yBC = slopeBC * mx2 + interceptBC;

    // find x coord in lines AB, AC, and BC, in the 2nd view for y=mean
    double xAB = (my2 - interceptAB) / slopeAB;
    double xAC = (my2 - interceptAC) / slopeAC;
    double xBC = (my2 - interceptBC) / slopeBC;

    // find min and max y and min and max x of thriangle vertices
    double min_y = std::min(std::min(tr_yA, tr_yB), tr_yC);
    double max_y = std::max(std::max(tr_yA, tr_yB), tr_yC);
    double min_x = std::min(std::min(tr_xA, tr_xB), tr_xC);
    double max_x = std::max(std::max(tr_xA, tr_xB), tr_xC);

    double max_distY;
    if (yAB>max_y || yAB<min_y) {
        max_distY = std::abs(yAC-yBC);
    } else if (yAC>max_y || yAC<min_y) {
        max_distY = std::abs(yAB-yBC);
    } else {
        max_distY = std::abs(yAB-yAC);
    }

    double max_distX;
    if (xAB>max_x || xAB<min_x) {
        max_distX = std::abs(xAC-xBC);
    } else if (xAC>max_x || xAC<min_x) {
        max_distX = std::abs(xAB-xBC);
    } else {
        max_distX = std::abs(xAB-xAC);
    }
    if (max_distX > max_distY){
        scale = max_distX;
        idx = 0;
    } else {
        scale = max_distY;
        idx = 1;
    }
}

void ThreeViewRelativePoseEstimator::estimate_models(std::vector<ThreeViewCameraPose> *models) {
    std::vector<CameraPose> models12;

    if (opt.use_affine) {
        switch (sample_sz) {
        case 4:
            relpose_affine_4p(x1, x2, sample, opt.max_epipolar_error * opt.max_epipolar_error, opt.early_nonminimal, &models12);
            break;
        case 3:
            affine_homography_3p(x1, x2, sample, opt.max_epipolar_error * opt.max_epipolar_error, opt.early_nonminimal, &models12);
            break;
        case 2:
            affine_essential_2p(x1, x2, sample, opt.max_epipolar_error * opt.max_epipolar_error, opt.early_nonminimal, &models12);
            break;
        }
    } else {
        relpose_5pt(x1n, x2n, &models12);

        if (opt.early_nonminimal){
            non_minimal_refinement(&models12);
        }

        if (opt.early_lm){
            lm_refinement(&models12);
        }
    }

    std::vector<Point3D> triangulated_12(3);

    for (CameraPose pose12 : models12){
//        std::vector<char> inliers;
//        int num_inliers = get_inliers(pose12, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, &inliers);
//        if (opt.use_affine)
//            std::cout << "Num inliers affine: " << num_inliers << std::endl;
//        else
//            std::cout << "Num inliers standard: " << num_inliers << std::endl;

        for (size_t i = 0; i < sample_sz_13; i++){
            triangulated_12[i] = triangulate(pose12, x1s[i], x2s[i]);
        }

        std::vector<CameraPose> models13;
        p3p(x3s, triangulated_12, &models13);

        for (CameraPose pose13 : models13){
            ThreeViewCameraPose three_view_pose = ThreeViewCameraPose(pose12, pose13);

            size_t inlier_13 = 0;
            size_t inlier_23 = 0;
            compute_sampson_msac_score(three_view_pose.pose13, x1, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_13);
            compute_sampson_msac_score(three_view_pose.pose23(), x2, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_23);

            if (opt.threeview_check and !opt.threeview_check_extra){
                size_t inlier_4p_13 = 0;
                size_t inlier_4p_23 = 0;
                std::vector<Point2D> x1c = {x1[sample[3]]};
                std::vector<Point2D> x2c = {x2[sample[3]]};
                std::vector<Point2D> x3c = {x3[sample[3]]};
                compute_sampson_msac_score(three_view_pose.pose13, x1c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_13);
                compute_sampson_msac_score(three_view_pose.pose23(), x2c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_23);
                if (inlier_4p_13 + inlier_4p_23 < 2) {
                    continue;
                }
            }

            if (opt.threeview_check and opt.threeview_check_extra){
                size_t inlier_4p_13 = 0;
                size_t inlier_4p_23 = 0;
                std::vector<Point2D> x1c = {x1[sample[3]], x1[sample[4]]};
                std::vector<Point2D> x2c = {x2[sample[3]], x2[sample[4]]};
                std::vector<Point2D> x3c = {x3[sample[3]], x3[sample[4]]};
                compute_sampson_msac_score(three_view_pose.pose13, x1c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_13);
                compute_sampson_msac_score(three_view_pose.pose23(), x2c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_23);
                if (inlier_4p_13 + inlier_4p_23 < 4) {
                    continue;
                }
            }


            if (opt.inner_refine > 0) {
                inner_refine(&three_view_pose);
            }

            models->emplace_back(three_view_pose);
        }
    }
}

void ThreeViewRelativePoseEstimator::inner_refine(ThreeViewCameraPose *three_view_pose) const {
    std::vector<Point2D> x1r, x2r, x3r;
    size_t n = 4 + opt.inner_refine_extra;
    x1r.resize(n);
    x2r.resize(n);
    x3r.resize(n);
    for (size_t k = 0; k < n; ++k) {
        x1r[k] = x1[sample[k]];
        x2r[k] = x2[sample[k]];
        x3r[k] = x3[sample[k]];
    }

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::CAUCHY;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.inner_refine;
    refine_3v_relpose(x1r, x2r, x3r, three_view_pose, bundle_opt);
}

double ThreeViewRelativePoseEstimator::score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const {
    size_t inlier_count12, inlier_count13, inlier_count23;
    // TODO: calc inliers better w/o redundant computation

    double score12 = compute_sampson_msac_score(three_view_pose.pose12, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count12);
    double score13 = compute_sampson_msac_score(three_view_pose.pose13, x1, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count13);
    double score23 = compute_sampson_msac_score(three_view_pose.pose23(), x2, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count23);

    std::vector<char> inliers;
    *inlier_count = get_inliers(three_view_pose, x1, x2, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inliers);
    return score12 + score13 + score23;
}

void ThreeViewRelativePoseEstimator::refine_model(ThreeViewCameraPose *pose) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;
//    bundle_opt.verbose = true;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*pose, x1, x2, x3, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier, x3_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);
    x3_inlier.reserve(num_inl);

    if (num_inl <= 4) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
            x3_inlier.push_back(x3[pt_k]);
        }
    }

//    size_t inliers_before;
//    double score_before = score_model(*pose, &inliers_before);
//    std::cout << "Score before: " << score_before <<std::endl;
//    std::cout << "Inliers before: " << inliers_before <<std::endl;

    refine_3v_relpose(x1_inlier, x2_inlier, x3_inlier, pose, bundle_opt);

//    size_t inliers_after;
//    double score_after = score_model(*pose, &inliers_after);
//    std::cout << "Score after: " << score_after <<std::endl;
//    std::cout << "Inliers after: " << inliers_after <<std::endl;
}

void ThreeViewSharedFocalRelativePoseEstimator::generate_nn_init_delta_models(std::vector<ImageTriplet> *models){
    std::vector<Eigen::Vector3f> P(5), PP(5);
    std::vector<Eigen::Vector3f> Q(5), QP(5);
    std::vector<Eigen::Vector3f> R(5), RP(5);

    for (size_t k = 0; k < 4; ++k){
        P[k] = x1[sample[k]].homogeneous().cast<float>();
        Q[k] = x2[sample[k]].homogeneous().cast<float>();
        R[k] = x3[sample[k]].homogeneous().cast<float>();
        PP[k] = x1[sample[k]].homogeneous().cast<float>();
        QP[k] = x2[sample[k]].homogeneous().cast<float>();
        RP[k] = x3[sample[k]].homogeneous().cast<float>();
    }

    P[4] = ((x1[sample[0]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().cast<float>();
    R[4] = ((x3[sample[0]] + x3[sample[1]] + x3[sample[2]]) / 3.0).homogeneous().cast<float>();
    PP[4] = ((x1[sample[3]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().cast<float>();
    RP[4] = ((x3[sample[3]] + x3[sample[1]] + x3[sample[2]]) / 3.0).homogeneous().cast<float>();

    std::vector<Eigen::Vector2f>  P1(5), Q1(5), R1(5), P1P(5), Q1P(5), R1P(5);
    Eigen::Matrix3f RP1, RQ1, RR1, RP1P, RQ1P, RR1P;

    Eigen::Vector2d q4 = (x2[sample[0]] + x2[sample[1]] + x2[sample[2]]) / 3.0;
    int idx;
    double scale;
    triangle_calc(q4(0), q4(1), 0, idx, scale);

    Eigen::Vector2d q4p = (x2[sample[3]] + x2[sample[1]] + x2[sample[2]]) / 3.0;
    int idxp;
    double scalep;
    triangle_calc(q4p(0), q4p(1), 3, idxp, scalep);

    for (size_t k = 0; k < 3; ++k){
        if (k == 0)
            Q[4] = q4.homogeneous().cast<float>();
        QP[4] = q4p.homogeneous().cast<float>();

        if (k == 1) {
            q4[idx] += scale * opt.delta;
            q4p[idx] += scalep * opt.delta;
            Q[4] = q4.homogeneous().cast<float>();
            QP[4] = q4p.homogeneous().cast<float>();
        }
        if (k == 2){
            q4[idx] -= 2 * scale * opt.delta;
            q4p[idx] -= 2 * scalep * opt.delta;
            Q[4] = q4.homogeneous().cast<float>();
            QP[4] = q4p.homogeneous().cast<float>();
        }

        normalize(P, Q, R, P1, Q1, R1, RP1, RQ1, RR1);
        normalize(PP, QP, RP, P1P, Q1P, R1P, RP1P, RQ1P, RR1P);

        float data[] = {P1[0][0], P1[0][1], Q1[0][0], Q1[0][1], R1[0][0], R1[0][1],
                        P1[1][0], P1[1][1], Q1[1][0], Q1[1][1], R1[1][0], R1[1][1],
                        P1[2][0], P1[2][1], Q1[2][0], Q1[2][1], R1[2][0], R1[2][1],
                        P1[3][0], P1[3][1], Q1[3][0], Q1[3][1], R1[3][0], R1[3][1]};

        std::vector<torch::jit::IValue> inputs;
        torch::Tensor f = torch::from_blob(data, {1, 4, 6});
        inputs.emplace_back(f);

        at::Tensor output_tensor;
        output_tensor = module.forward(inputs).toTensor();
        Eigen::Vector3f pred1, pred2;
        pred1 << output_tensor[0][0].item<float>(), output_tensor[0][1].item<float>(), 1.0;
        pred1 = pred1 / pred1.norm();
        Eigen::Vector3f rp = RQ1.transpose() * pred1;
        x2n[5] = rp.normalized().cast<double>();


        float datap[] = {P1P[0][0], P1P[0][1], Q1P[0][0], Q1P[0][1], R1P[0][0], R1P[0][1],
                         P1P[1][0], P1P[1][1], Q1P[1][0], Q1P[1][1], R1P[1][0], R1P[1][1],
                         P1P[2][0], P1P[2][1], Q1P[2][0], Q1P[2][1], R1P[2][0], R1P[2][1],
                         P1P[3][0], P1P[3][1], Q1P[3][0], Q1P[3][1], R1P[3][0], R1P[3][1]};

        std::vector<torch::jit::IValue> inputsp;
        torch::Tensor fp = torch::from_blob(datap, {1, 4, 6});
        inputsp.emplace_back(fp);

        at::Tensor output_tensorp;
        output_tensorp = module.forward(inputsp).toTensor();
        Eigen::Vector3f pred1p, pred2p;
        pred1p << output_tensorp[0][0].item<float>(), output_tensorp[0][1].item<float>(), 1.0;
        pred1p = pred1p / pred1p.norm();
        Eigen::Vector3f rpp = RQ1P.transpose() * pred1p;
        x2n[4] = rpp.normalized().cast<double>();

        estimate_models(models);
    }
}

Eigen::Vector3d ThreeViewSharedFocalRelativePoseEstimator::get_network_point(int idx_t){
    std::vector<Eigen::Vector3f> P(5);
    std::vector<Eigen::Vector3f> Q(5);
    std::vector<Eigen::Vector3f> R(5);

    for (size_t k = 0; k < 4; ++k){
        P[k] = x1[sample[k]].homogeneous().cast<float>();
        Q[k] = x2[sample[k]].homogeneous().cast<float>();
        R[k] = x3[sample[k]].homogeneous().cast<float>();
    }
    P[4] = ((x1[sample[idx_t]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().cast<float>();
    Q[4] = ((x2[sample[idx_t]] + x2[sample[1]] + x2[sample[2]]) / 3.0).homogeneous().cast<float>();
    R[4] = ((x3[sample[idx_t]] + x3[sample[1]] + x3[sample[2]]) / 3.0).homogeneous().cast<float>();

    std::vector<Eigen::Vector2f>  P1(5), Q1(5), R1(5);
    Eigen::Matrix3f RP1, RQ1, RR1;

    normalize(P, Q, R, P1, Q1, R1, RP1, RQ1, RR1);

//    std::cout << "RR1" << std::endl << RR1 << std::endl;

    float data[] = {P1[0][0], P1[0][1], Q1[0][0], Q1[0][1], R1[0][0], R1[0][1],
                    P1[1][0], P1[1][1], Q1[1][0], Q1[1][1], R1[1][0], R1[1][1],
                    P1[2][0], P1[2][1], Q1[2][0], Q1[2][1], R1[2][0], R1[2][1],
                    P1[3][0], P1[3][1], Q1[3][0], Q1[3][1], R1[3][0], R1[3][1]};

    std::vector<torch::jit::IValue> inputs;
    torch::Tensor f = torch::from_blob(data, {1, 4, 6});
    inputs.push_back(f);

    at::Tensor output_tensor;
    output_tensor = module.forward(inputs).toTensor();
    // vectorize prediction
    Eigen::Vector3f pred1, pred2;

    pred1 << output_tensor[0][0].item<float>(), output_tensor[0][1].item<float>(), 1.0;
    pred1 = pred1 / pred1.norm();
    Eigen::Vector3f rp = RQ1.transpose() * pred1;
    return rp.normalized().cast<double>();
}

void ThreeViewSharedFocalRelativePoseEstimator::generate_models(std::vector<ImageTriplet> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1n[k] = x1[sample[k]].homogeneous().normalized();
        x2n[k] = x2[sample[k]].homogeneous().normalized();
    }

    for (size_t k = 0; k < sample_sz_13; ++k) {
        x1s[k] = x1[sample[k]].homogeneous();
        x2s[k] = x2[sample[k]].homogeneous();
    }

    if (opt.init_net){
        x1n[5] = ((x1[sample[0]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().normalized();
        x1n[4] = ((x1[sample[3]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().normalized();
        generate_nn_init_delta_models(models);
        return;
    }

    if (opt.oracle){
        if (sample_sz < 6){
            x1n[5] = ((x1[sample[0]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().normalized();
            Eigen::Vector3d epipolar_line = (opt.gt_E * x1n[5]).normalized();

            double x_0 = (x2[sample[0]](0) + x2[sample[1]](0) + x2[sample[2]](0)) / 3.0;
            double y = - (epipolar_line(2) + epipolar_line(0) * x_0) / epipolar_line(1);
            x2n[5] = Eigen::Vector2d(x_0, y).homogeneous().normalized();
        }

        if (sample_sz < 5) {
            x1n[4] = ((x1[sample[3]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().normalized();
            Eigen::Vector3d epipolar_line = (opt.gt_E * x1n[4]).normalized();
            double x_0 = (x2[sample[3]](0) + x2[sample[1]](0) + x2[sample[2]](0)) / 3.0;
            double y = -(epipolar_line(2) + epipolar_line(0) * x_0) / epipolar_line(1);
            x2n[4] = Eigen::Vector2d(x_0, y).homogeneous().normalized();
        }
        estimate_models(models);
        return;
    }

    // if we use 5 pts we approx the last as the mean of the first 3 pts
    if (sample_sz < 6){
        x1n[5] = ((x1[sample[0]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().normalized();
        if (opt.use_net){
            x2n[5] = get_network_point(0);
        } else {
            x2n[5] = ((x2[sample[0]] + x2[sample[1]] + x2[sample[2]]) / 3.0).homogeneous().normalized();

        }
    }

    if (sample_sz < 5){
        x1n[4] = ((x1[sample[3]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().normalized();
        if (opt.use_net) {
            x2n[4] = get_network_point(3);
        } else {
            x2n[4] = ((x2[sample[3]] + x2[sample[1]] + x2[sample[2]]) / 3.0).homogeneous().normalized();
        }
    }

    estimate_models(models);

    if (sample_sz < 6 and opt.delta > 0.0){
        delta(models);
    }
}

void ThreeViewSharedFocalRelativePoseEstimator::delta(std::vector<ImageTriplet> *models) {
    double mx = x2n[5](0) / x2n[5](1), my = x2n[5](1) / x2n[5](1);

    int idx;
    double scale;
    triangle_calc(mx, my, 0, idx, scale);

    if (sample_sz == 5) {
        Eigen::Vector2d x2n5(mx, my);
        x2n5(idx) += opt.delta * scale;
        x2n[5] = x2n5.homogeneous().normalized();
        estimate_models(models);

        x2n5(idx) -= 2 * opt.delta * scale;
        x2n[5] = x2n5.homogeneous().normalized();
        estimate_models(models);
        return;
    }

    if (sample_sz == 4) {
        double mxp = x2n[4](0) / x2n[4](2), myp = x2n[4](1) / x2n[4](2);

        int idxp;
        double scalep;
        triangle_calc(mxp, myp, 3, idxp, scalep);

        Eigen::Vector2d x2n5(mx, my);
        Eigen::Vector2d x2n4(mxp, myp);
        x2n5(idx) += opt.delta * scale;
        x2n4(idxp) += opt.delta * scalep;
        x2n[5] = x2n5.homogeneous().normalized();
        x2n[4] = x2n4.homogeneous().normalized();
        estimate_models(models);

        x2n5(idx) -= 2 * opt.delta * scale;
        x2n4(idxp) -= 2 * opt.delta * scalep;
        x2n[5] = x2n5.homogeneous().normalized();
        x2n[4] = x2n4.homogeneous().normalized();
        estimate_models(models);
        return;
    }
}

void ThreeViewSharedFocalRelativePoseEstimator::estimate_models(std::vector<ImageTriplet> *models) {
    std::vector<ImagePair> models12;
    relpose_6pt_focal(x1n, x2n, &models12);

    std::vector<Point3D> triangulated_12;
    triangulated_12.reserve(3);

    for (ImagePair pair12 : models12){
        for (size_t i = 0; i < sample_sz_13; i++){
            triangulated_12[i] = triangulate(pair12, x1s[i], x2s[i]);
        }

        for (size_t k = 0; k < sample_sz_13; ++k) {
            Eigen::Vector2d x;
            pair12.camera1.unproject(x3[sample[k]], &x);
            x3s[k] = x.homogeneous().normalized();
        }

        std::vector<CameraPose> models13;

        p3p(x3s, triangulated_12, &models13);

        for (CameraPose pose13 : models13){
            ImageTriplet image_triplet = ImageTriplet(ThreeViewCameraPose(pair12.pose, pose13), pair12.camera1);
//            std::cout << "R: " << std::endl << pose13.R() << std::endl;
//            std::cout << "t: " << pose13.t.transpose() << std::endl;
            if (opt.threeview_check and !opt.threeview_check_extra){
                size_t inlier_4p_13 = 0;
                size_t inlier_4p_23 = 0;
                std::vector<Point2D> x1c = {x1[sample[3]]};
                std::vector<Point2D> x2c = {x2[sample[3]]};
                std::vector<Point2D> x3c = {x3[sample[3]]};

                Eigen::DiagonalMatrix<double, 3> K_inv(1, 1, image_triplet.camera.focal());
                Eigen::Matrix3d F13, F23;
                essential_from_motion(image_triplet.poses.pose13, &F13);
                essential_from_motion(image_triplet.poses.pose23(), &F23);
                F13 = K_inv * F13 * K_inv;
                F23 = K_inv * F23 * K_inv;

                compute_sampson_msac_score(F13, x1c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_13);
                compute_sampson_msac_score(F23, x2c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_23);
                if (inlier_4p_13 + inlier_4p_23 < 2) {
                    continue;
                }
            }
            if (opt.threeview_check and opt.threeview_check_extra){
                size_t inlier_4p_13 = 0;
                size_t inlier_4p_23 = 0;
                std::vector<Point2D> x1c = {x1[sample[3]], x1[sample[4]]};
                std::vector<Point2D> x2c = {x2[sample[3]], x2[sample[4]]};
                std::vector<Point2D> x3c = {x3[sample[3]], x3[sample[4]]};

                Eigen::DiagonalMatrix<double, 3> K_inv(1, 1, image_triplet.camera.focal());
                Eigen::Matrix3d F13, F23;
                essential_from_motion(image_triplet.poses.pose13, &F13);
                essential_from_motion(image_triplet.poses.pose23(), &F23);
                F13 = K_inv * F13 * K_inv;
                F23 = K_inv * F23 * K_inv;

                compute_sampson_msac_score(F13, x1c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_13);
                compute_sampson_msac_score(F23, x2c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_23);
                if (inlier_4p_13 + inlier_4p_23 < 4) {
                    continue;
                }
            }

            if (opt.inner_refine > 0) {
                inner_refine(&image_triplet);
            }
            models->emplace_back(image_triplet);
        }
    }
}

void ThreeViewSharedFocalRelativePoseEstimator::inner_refine(ImageTriplet *image_triplet) const {
    std::vector<Point2D> x1r, x2r, x3r;
    size_t n = 4 + opt.inner_refine_extra;
    x1r.resize(n);
    x2r.resize(n);
    x3r.resize(n);
    for (size_t k = 0; k < n; ++k) {
        x1r[k] = x1[sample[k]];
        x2r[k] = x2[sample[k]];
        x3r[k] = x3[sample[k]];
    }

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::CAUCHY;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.inner_refine;
    refine_3v_shared_focal_relpose(x1r, x2r, x3r, image_triplet, bundle_opt);
}

double ThreeViewSharedFocalRelativePoseEstimator::score_model(const ImageTriplet &image_triplet, size_t *inlier_count) const {
    size_t inlier_count12, inlier_count13, inlier_count23;
    // TODO: calc inliers better w/o redundant computation

    Eigen::DiagonalMatrix<double, 3> K_inv(1, 1, image_triplet.camera.focal());
    Eigen::Matrix3d F12, F13, F23;
    essential_from_motion(image_triplet.poses.pose12, &F12);
    essential_from_motion(image_triplet.poses.pose13, &F13);
    essential_from_motion(image_triplet.poses.pose23(), &F23);
    F12 = K_inv * F12 * K_inv;
    F13 = K_inv * F13 * K_inv;
    F23 = K_inv * F23 * K_inv;
    double score12 = compute_sampson_msac_score(F12, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count12);
    double score13 = compute_sampson_msac_score(F13, x1, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count13);
    double score23 = compute_sampson_msac_score(F23, x2, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count23);

    std::vector<char> inliers;
    *inlier_count = get_inliers(image_triplet, x1, x2, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inliers);
    return score12 + score13 + score23;
}

void ThreeViewSharedFocalRelativePoseEstimator::refine_model(ImageTriplet *image_triplet) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;
//    bundle_opt.verbose = true;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*image_triplet, x1, x2, x3, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier, x3_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);
    x3_inlier.reserve(num_inl);

    if (num_inl <= 4) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
            x3_inlier.push_back(x3[pt_k]);
        }
    }

//    size_t inliers_before;
//    double score_before = score_model(*image_triplet, &inliers_before);
//    std::cout << "Score before: " << score_before <<std::endl;
//    std::cout << "Inliers before: " << inliers_before <<std::endl;

    refine_3v_shared_focal_relpose(x1_inlier, x2_inlier, x3_inlier, image_triplet, bundle_opt);

//    size_t inliers_after;
//    double score_after = score_model(*image_triplet, &inliers_after);
//    std::cout << "Score after: " << score_after <<std::endl;
//    std::cout << "Inliers after: " << inliers_after <<std::endl;
}

void ThreeViewSharedFocalRelativePoseEstimator::triangle_calc(double mx, double my, int idx_t, int &idx, double &scale) {
    double tr_xA = x2[sample[idx_t]](0), tr_yA = x2[sample[idx_t]](1);
    double tr_xB = x2[sample[1]](0), tr_yB = x2[sample[1]](1);
    double tr_xC = x2[sample[2]](0), tr_yC = x2[sample[2]](1);

    double slopeAB = (tr_yB - tr_yA) / (tr_xB - tr_xA);
    double slopeAC = (tr_yC - tr_yA) / (tr_xC - tr_xA);
    double slopeBC = (tr_yC - tr_yB) / (tr_xC - tr_xB);

    double interceptAB = tr_yA - slopeAB * tr_xA;
    double interceptAC = tr_yA - slopeAC * tr_xA;
    double interceptBC = tr_yB - slopeBC * tr_xB;

    // find y coord in lines AB, AC, and BC, in the 2nd view for x=mean
    double yAB = slopeAB * mx + interceptAB;
    double yAC = slopeAC * mx + interceptAC;
    double yBC = slopeBC * mx + interceptBC;

    // find x coord in lines AB, AC, and BC, in the 2nd view for y=mean
    double xAB = (my - interceptAB) / slopeAB;
    double xAC = (my - interceptAC) / slopeAC;
    double xBC = (my - interceptBC) / slopeBC;

    // find min and max y and min and max x of thriangle vertices
    double min_y = std::min(std::min(tr_yA, tr_yB), tr_yC);
    double max_y = std::max(std::max(tr_yA, tr_yB), tr_yC);
    double min_x = std::min(std::min(tr_xA, tr_xB), tr_xC);
    double max_x = std::max(std::max(tr_xA, tr_xB), tr_xC);

    double max_distY;
    if (yAB>max_y || yAB<min_y) {
        max_distY = std::abs(yAC-yBC);
    } else if (yAC>max_y || yAC<min_y) {
        max_distY = std::abs(yAB-yBC);
    } else {
        max_distY = std::abs(yAB-yAC);
    }

    double max_distX;
    if (xAB>max_x || xAB<min_x) {
        max_distX = std::abs(xAC-xBC);
    } else if (xAC>max_x || xAC<min_x) {
        max_distX = std::abs(xAB-xBC);
    } else {
        max_distX = std::abs(xAB-xAC);
    }
    if (max_distX > max_distY){
        scale = max_distX;
        idx = 0;
    } else {
        scale = max_distY;
        idx = 1;
    }
}

int ThreeViewSharedFocalRelativePoseEstimator::normalize(std::vector<Eigen::Vector3f> &P, std::vector<Eigen::Vector3f> &Q,
                                                  std::vector<Eigen::Vector3f> &T, std::vector<Eigen::Vector2f> &P1,
                                                  std::vector<Eigen::Vector2f> &Q1, std::vector<Eigen::Vector2f> &T1,
                                                  Eigen::Matrix3f &CP1, Eigen::Matrix3f &CQ1, Eigen::Matrix3f &CT1) const
    {
//    Eigen::Vector3f centroidP = Eigen::Vector3f::Zero();
//    Eigen::Vector3f centroidQ = Eigen::Vector3f::Zero();
        for(int i=0;i<5;++i){
            P[i] = P[i]/P[i].norm();
            Q[i] = Q[i]/Q[i].norm();
            T[i] = T[i]/T[i].norm();
        }

        // generate the rotation matrix that sends first point to (0, 0) and second point to (0, x)

        // for view #1
        Eigen::Vector3f p0 = P[4]/P[4].norm();
        Eigen::Vector3f p1 = p0.cross(P[3]);
        p1 = p1/p1.norm();
        Eigen::Vector3f p2 = p0.cross(p1);
        p2 = p2/p2.norm();
        Eigen::Matrix3f Zp;
        Zp << p0, p1, p2;

        // for view #2
        Eigen::Vector3f q0 = Q[4]/Q[4].norm();
        Eigen::Vector3f q1 = q0.cross(Q[3]);
        q1 = q1/q1.norm();
        Eigen::Vector3f q2 = q0.cross(q1);
        q2 = q2/q2.norm();
        Eigen::Matrix3f Zq;
        Zq << q0, q1, q2;

        // for view #3
        Eigen::Vector3f t0 = T[4]/T[4].norm();
        Eigen::Vector3f t1 = t0.cross(T[3]);
        t1 = t1/t1.norm();
        Eigen::Vector3f t2 = t0.cross(t1);
        t2 = t2/t2.norm();
        Eigen::Matrix3f Zt;
        Zt << t0, t1, t2;

        Eigen::Matrix3f ZZ;
        ZZ << 0,-1,0,0,0,-1,1,0,0;
        Eigen::Matrix3f CP = ZZ * Zp.transpose();
        Eigen::Matrix3f CQ = ZZ * Zq.transpose();
        Eigen::Matrix3f CT = ZZ * Zt.transpose();

        // rotate the points and project them back to the plane
        for(int i=0;i<5;++i){
            P[i] = CP * P[i];
            P[i] = P[i]/P[i](2);

            Q[i] = CQ * Q[i];
            Q[i] = Q[i]/Q[i](2);

            T[i] = CT * T[i];
            T[i] = T[i]/T[i](2);
        }

        // hardcode the "0" values because of the limitation of numerical precision
        //P[0][0] = 0.0;
        //P[0][1] = 0.0;
        //P[1][0] = 0.0;
        P[3][0] = 0.0;

        //Q[0][0] = 0.0;
        //Q[0][1] = 0.0;
        //Q[1][0] = 0.0;
        Q[3][0] = 0.0;

        //T[0][0] = 0.0;
        //T[0][1] = 0.0;
        //T[1][0] = 0.0;
        T[3][0] = 0.0;

        for(int i=0;i<5;++i){
            Eigen::Vector2f p;
            p(0) = P[i](0);
            p(1) = P[i](1);
            P1[i] = p;

            Eigen::Vector2f q;
            q(0) = Q[i](0);
            q(1) = Q[i](1);
            Q1[i] = q;

            Eigen::Vector2f t;
            t(0) = T[i](0);
            t(1) = T[i](1);
            T1[i] = t;
        }
        CP1 = CP;
        CQ1 = CQ;
        CT1 = CT;

        return 1;
    }

void SharedFocalRelativePoseEstimator::generate_models(ImagePairVector *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_6pt_focal(x1s, x2s, models);
}

double SharedFocalRelativePoseEstimator::score_model(const ImagePair &image_pair, size_t *inlier_count) const {
    Eigen::Matrix3d K_inv;
    K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, image_pair.camera1.focal();
    // K_inv << 1.0 / calib_pose.camera.focal(), 0.0, 0.0, 0.0, 1.0 / calib_pose.camera.focal(), 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d E;
    essential_from_motion(image_pair.pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void SharedFocalRelativePoseEstimator::refine_model(ImagePair *image_pair) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    Eigen::Matrix3d K_inv;
    // K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, calib_pose->camera.focal();
    K_inv << 1.0 / image_pair->camera1.focal(), 0.0, 0.0, 0.0, 1.0 / image_pair->camera1.focal(), 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d E;
    essential_from_motion(image_pair->pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(F, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);

    if (num_inl <= 6) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
        }
    }

    refine_shared_focal_relpose(x1_inlier, x2_inlier, image_pair, bundle_opt);
}

void GeneralizedRelativePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    // TODO replace by general 6pt solver?

    bool done = false;
    int pair0 = 0, pair1 = 1;
    while (!done) {
        pair0 = random_int(rng) % matches.size();
        if (matches[pair0].x1.size() < 5)
            continue;

        pair1 = random_int(rng) % matches.size();
        if (pair0 == pair1 || matches[pair1].x1.size() == 0)
            continue;

        done = true;
    }

    // Sample 5 points from the first camera pair
    CameraPose pose1 = rig1_poses[matches[pair0].cam_id1];
    CameraPose pose2 = rig2_poses[matches[pair0].cam_id2];
    Eigen::Vector3d p1 = pose1.center();
    Eigen::Vector3d p2 = pose2.center();
    draw_sample(5, matches[pair0].x1.size(), &sample, rng);
    for (size_t k = 0; k < 5; ++k) {
        x1s[k] = pose1.derotate(matches[pair0].x1[sample[k]].homogeneous().normalized());
        p1s[k] = p1;
        x2s[k] = pose2.derotate(matches[pair0].x2[sample[k]].homogeneous().normalized());
        p2s[k] = p2;
    }

    // Sample one point from the second camera pair
    pose1 = rig1_poses[matches[pair1].cam_id1];
    pose2 = rig2_poses[matches[pair1].cam_id2];
    p1 = pose1.center();
    p2 = pose2.center();
    size_t ind = random_int(rng) % matches[pair1].x1.size();
    x1s[5] = pose1.derotate(matches[pair1].x1[ind].homogeneous().normalized());
    p1s[5] = p1;
    x2s[5] = pose2.derotate(matches[pair1].x2[ind].homogeneous().normalized());
    p2s[5] = p2;

    gen_relpose_5p1pt(p1s, x1s, p2s, x2s, models);
}

double GeneralizedRelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    *inlier_count = 0;
    double cost = 0;
    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose.t);
        pose2.q = quat_multiply(pose2.q, pose.q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        size_t local_inlier_count = 0;
        cost += compute_sampson_msac_score(relpose, m.x1, m.x2, opt.max_epipolar_error * opt.max_epipolar_error,
                                           &local_inlier_count);
        *inlier_count += local_inlier_count;
    }

    return cost;
}

void GeneralizedRelativePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    std::vector<PairwiseMatches> inlier_matches;
    inlier_matches.resize(matches.size());

    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose->t);
        pose2.q = quat_multiply(pose2.q, pose->q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        // Compute inliers with a relaxed threshold
        std::vector<char> inliers;
        int num_inl = get_inliers(relpose, m.x1, m.x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);

        inlier_matches[match_k].cam_id1 = m.cam_id1;
        inlier_matches[match_k].cam_id2 = m.cam_id2;
        inlier_matches[match_k].x1.reserve(num_inl);
        inlier_matches[match_k].x2.reserve(num_inl);

        for (size_t k = 0; k < m.x1.size(); ++k) {
            if (inliers[k]) {
                inlier_matches[match_k].x1.push_back(m.x1[k]);
                inlier_matches[match_k].x2.push_back(m.x2[k]);
            }
        }
    }

    refine_generalized_relpose(inlier_matches, rig1_poses, rig2_poses, pose, bundle_opt);
}

void FundamentalEstimator::generate_models(std::vector<Eigen::Matrix3d> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_7pt(x1s, x2s, models);

    if (opt.real_focal_check) {
        for (int i = models->size() - 1; i >= 0; i--) {
            if (!calculate_RFC((*models)[i]))
                models->erase(models->begin() + i);
        }
    }
}

double FundamentalEstimator::score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const {
    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void FundamentalEstimator::refine_model(Eigen::Matrix3d *F) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    refine_fundamental(x1, x2, F, bundle_opt);
}

} // namespace poselib
