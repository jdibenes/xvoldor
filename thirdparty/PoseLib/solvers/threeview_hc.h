//
// Created by kocur on 26-Jun-24.
//

#ifndef POSELIB_THREEVIEW_HC_H
#define POSELIB_THREEVIEW_HC_H

#include <iostream>
#include "PoseLib/camera_pose.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/types.h"

namespace poselib{

class ThreeViewRelativePoseHCEstimator {
    public:
        ThreeViewRelativePoseHCEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                         const std::vector<Point2D> &points2D_2, const std::vector<Point2D> &points2D_3                                   )
                : num_data(points2D_1.size()), opt(ransac_opt), xx1(points2D_1), xx2(points2D_2), xx3(points2D_3),
                  sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
            x1n.resize(sample_sz);
            x2n.resize(sample_sz);
            x3n.resize(sample_sz);
            sample.resize(sample_sz);
            std::string anchorfile("res/anchors.txt");
            bool anchors_loaded = load_anchors(anchorfile);
            if (!anchors_loaded)
                std::cout << "Failed to load anchors" << std::endl;
//            assert(anchors_loaded);
            // Loads the neural network parameters.
            std::string nn_file("res/nn.txt");
            bool nn_loaded = load_NN(nn_file);
            if (!nn_loaded)
                std::cout << "Failed to load NN" << std::endl;
//            assert(nn_loaded);
        }

    void generate_models(std::vector<ThreeViewCameraPose> *models);
    double score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const;
    void refine_model(ThreeViewCameraPose *three_view_pose) const;

    const size_t sample_sz = 4;
    const size_t num_data;

protected:
    // data for the anchors
    std::vector<std::vector<double>> problems_anchors;
    std::vector<std::vector<double>> start_anchors;
    std::vector<std::vector<double>> depths_anchors;
    // data for the neural network
    int num_layers;
    std::vector<std::vector<float>> ws;
    std::vector<std::vector<float>> bs;
    std::vector<std::vector<float>> ps;
    std::vector<int> a_;
    std::vector<int> b_;

    bool load_anchors(const std::string& data_file);
    bool load_NN(const std::string& nn_file);
    void order_points(std::vector<Eigen::Vector3d> &P, int * perm4, int ix) const;
    void normalize(std::vector<Eigen::Vector3d> &P,std::vector<Eigen::Vector3d> &Q,std::vector<Eigen::Vector3d> &R,
                   std::vector<Eigen::Vector2d> &P1, std::vector<Eigen::Vector2d> &Q1, std::vector<Eigen::Vector2d> &R1,
                   Eigen::Matrix3d &CP1, Eigen::Matrix3d &CQ1, Eigen::Matrix3d &CR1, int * perm, int * perm4) const;
    void extract_pose12(double params[24], double solution[12], Eigen::Matrix3d &R, Eigen::Vector3d &t) const;
    void extract_pose13(double params[24], double solution[12], Eigen::Matrix3d &R, Eigen::Vector3d &t) const;

private:
    const RansacOptions &opt;
    const std::vector<Point2D> &xx1;
    const std::vector<Point2D> &xx2;
    const std::vector<Point2D> &xx3;
    std::vector<Point3D> x1n, x2n, x3n;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<size_t> sample;
};
} // namespace poselib

#endif //POSELIB_THREEVIEW_HC_H
