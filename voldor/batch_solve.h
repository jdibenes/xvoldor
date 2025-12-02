
#pragma once

#include <opencv2/calib3d.hpp>

int batch_solve_ap3p_cpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
int batch_solve_lambdatwist_cpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
int batch_solve_ap3p_gpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
int batch_solve_lambdatwist_gpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);

int batch_solve_tft_linear_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, std::vector<cv::Point3f> const& pts2, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);

int batch_solve_gpm_hpc0_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
int batch_solve_gpm_hpc1_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool, int refine_iterations);
int batch_solve_gpm_hpc2_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool, int refine_iterations);
