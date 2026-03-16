
#pragma once

#include <opencv2/calib3d.hpp>

int batch_cpu_solver_p4p_lambdatwist(cv::Point2f const* p2d, cv::Point3f const* p3d, int point_count, cv::Mat const& K, int poses_to_sample, float* poses, int workers);
int batch_cpu_solver_rnp(cv::Point2f const* p2d, cv::Point3f const* p3d, int point_count, int sample_size, cv::Mat const& K, int solver, bool direction, float r0, int max_pow, int max_iterations, int poses_to_sample, float* poses, int workers);


int batch_solve_ap3p_cpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
//int batch_solve_lambdatwist_cpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
int batch_solve_ap3p_gpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
int batch_solve_lambdatwist_gpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);

int batch_solve_tft_linear_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, std::vector<cv::Point3f> const& pts2, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool, std::vector<cv::Vec6f>* next_pool);

int batch_solve_gpm_hpc0_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
int batch_solve_gpm_hpc1_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool, int refine_iterations);
int batch_solve_gpm_hpc2_cpu(std::vector<cv::Point3f> const& pts0, std::vector<cv::Point3f> const& pts1, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool, int refine_iterations);
