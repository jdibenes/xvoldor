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

#ifndef POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H
#define POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include "PoseLib/camera_pose.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/types.h"
#include <iostream>

namespace poselib {

class RelativePoseEstimator {
  public:
    RelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                          const std::vector<Point2D> &points2D_2)
        : sample_sz((ransac_opt.sample_sz == 0) ? 5 : ransac_opt.sample_sz),
          num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(ransac_opt.use_homography ? 4 : 5);
        x2s.resize(ransac_opt.use_homography ? 4 : 5);
        sample.resize(sample_sz);
    }

    Point2D triangle_calc();
    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;
    void non_minimal_refinement(std::vector<CameraPose> *models) const;

    const size_t sample_sz;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    Point2D m1, m2;

    RandomSampler sampler;

    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
    void pose_from_H(Eigen::Matrix3d &H, std::vector<CameraPose> *models) const;
    void lm_refinement(std::vector<CameraPose> *models) const;
};

class ThreeViewRelativePoseEstimator {
  public:
    ThreeViewRelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                   const std::vector<Point2D> &points2D_2, const std::vector<Point2D> &points2D_3)
        : sample_sz((ransac_opt.sample_sz == 0) ? 5 : ransac_opt.sample_sz), num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), x3(points2D_3),
        sampler(num_data, (ransac_opt.sample_sz > 2) ? sample_sz : 3, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        // we do not use virtual points with affine
        x1n.resize(opt.use_affine ? sample_sz : 5);
        x2n.resize(opt.use_affine ? sample_sz : 5);
        x1s.resize(sample_sz_13);
        x2s.resize(sample_sz_13);
        x3s.resize(sample_sz_13);
        sample.resize(sample_sz);
        if (opt.use_net || opt.init_net){
            module = torch::jit::load("res/epoch28_sampson.pt");
        }

        if (opt.use_nister > 0){
            epipole = opt.gt_E.fullPivLu().kernel().col(0).hnormalized().homogeneous();
        }
    }

    void generate_models(std::vector<ThreeViewCameraPose> *models);
    double score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const;
    void refine_model(ThreeViewCameraPose *three_view_pose) const;
    void inner_refine(ThreeViewCameraPose *three_view_pose) const;

    const size_t sample_sz;
    const size_t sample_sz_13 = 3;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;
    torch::jit::script::Module module;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1n, x2n, x1s, x2s, x3s;
    std::vector<size_t> sample;
    Eigen::Vector3d epipole;

    void estimate_models(std::vector<ThreeViewCameraPose> *models);
    void triangle_calc(double mx, double my, int &idx, double &scale);
    void delta(double mx, double my, std::vector<ThreeViewCameraPose> *models);
    int normalize(std::vector<Eigen::Vector3f> &P, std::vector<Eigen::Vector3f> &Q, std::vector<Eigen::Vector2f> &P1,
                   std::vector<Eigen::Vector2f> &Q1, Eigen::Matrix3f &CP1, Eigen::Matrix3f &CQ1) const;

    int normalize(std::vector<Eigen::Vector3f> &P, std::vector<Eigen::Vector3f> &Q, std::vector<Eigen::Vector3f> &T,
                  std::vector<Eigen::Vector2f> &P1, std::vector<Eigen::Vector2f> &Q1, std::vector<Eigen::Vector2f> &T1,
                  Eigen::Matrix3f &CP1, Eigen::Matrix3f &CQ1, Eigen::Matrix3f &CT1) const;

    Eigen::Vector3d get_network_point();
    void generate_nn_init_delta_models(std::vector<ThreeViewCameraPose> *models);
    void non_minimal_refinement(std::vector<CameraPose> *models) const;
    void lm_refinement(std::vector<CameraPose> *models) const;
};

class ThreeViewSharedFocalRelativePoseEstimator {
  public:
    ThreeViewSharedFocalRelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                              const std::vector<Point2D> &points2D_2,
                                              const std::vector<Point2D> &points2D_3)
        : sample_sz((ransac_opt.sample_sz == 0) ? 6 : ransac_opt.sample_sz), num_data(points2D_1.size()),
          opt(ransac_opt), x1(points2D_1), x2(points2D_2), x3(points2D_3),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1n.resize(6);
        x2n.resize(6);
        x1s.resize(sample_sz_13);
        x2s.resize(sample_sz_13);
        x3s.resize(sample_sz_13);
        sample.resize(sample_sz);
        if (opt.use_net || opt.init_net){
            module = torch::jit::load("res/epoch28_sampson.pt");
        }
    }

    void generate_models(std::vector<ImageTriplet> *models);
    double score_model(const ImageTriplet &image_triplet, size_t *inlier_count) const;
    void refine_model(ImageTriplet *image_triplet) const;

    const size_t sample_sz;
    const size_t sample_sz_13 = 3;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;
    torch::jit::script::Module module;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1n, x2n, x1s, x2s, x3s;
    std::vector<size_t> sample;

    void estimate_models(std::vector<ImageTriplet> *models);
    void delta(std::vector<ImageTriplet> *models);

    void triangle_calc(double mx, double my, int idx_t, int &idx, double &scale);

    void inner_refine(ImageTriplet *image_triplet) const;

    Eigen::Vector3d get_network_point(int idx_t);

    int normalize(std::vector<Eigen::Vector3f> &P, std::vector<Eigen::Vector3f> &Q, std::vector<Eigen::Vector3f> &T,
                  std::vector<Eigen::Vector2f> &P1, std::vector<Eigen::Vector2f> &Q1, std::vector<Eigen::Vector2f> &T1,
                  Eigen::Matrix3f &CP1, Eigen::Matrix3f &CQ1, Eigen::Matrix3f &CT1) const;
    void generate_nn_init_delta_models(std::vector<ImageTriplet> *models);
};

class SharedFocalRelativePoseEstimator {
  public:
    SharedFocalRelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                     const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(ImagePairVector *models);
    double score_model(const ImagePair &image_pair, size_t *inlier_count) const;
    void refine_model(ImagePair *image_pair) const;

    const size_t sample_sz = 6;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class GeneralizedRelativePoseEstimator {
  public:
    GeneralizedRelativePoseEstimator(const RansacOptions &ransac_opt,
                                     const std::vector<PairwiseMatches> &pairwise_matches,
                                     const std::vector<CameraPose> &camera1_ext,
                                     const std::vector<CameraPose> &camera2_ext)
        : opt(ransac_opt), matches(pairwise_matches), rig1_poses(camera1_ext), rig2_poses(camera2_ext) {
        rng = opt.seed;
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        p1s.resize(sample_sz);
        p2s.resize(sample_sz);
        sample.resize(sample_sz);

        num_data = 0;
        for (const PairwiseMatches &m : matches) {
            num_data += m.x1.size();
        }
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 6;
    size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &rig1_poses;
    const std::vector<CameraPose> &rig2_poses;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s, p1s, p2s;
    std::vector<size_t> sample;
};

class FundamentalEstimator {
  public:
    FundamentalEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                         const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<Eigen::Matrix3d> *models);
    double score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const;
    void refine_model(Eigen::Matrix3d *F) const;

    const size_t sample_sz = 7;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

} // namespace poselib

#endif