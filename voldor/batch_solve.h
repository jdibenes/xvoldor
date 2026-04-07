
#pragma once

#include <opencv2/calib3d.hpp>

int batch_cpu_solver_p4p_lambdatwist(cv::Point2f const* p2d, cv::Point3f const* p3d, int point_count, cv::Mat const& K, int poses_to_sample, float* poses, int workers, bool unique);


int batch_cpu_solver_gpm(cv::Point3f const* p3d_1, cv::Point3f const* p3d_2, int point_count, int solver, int poses_to_sample, float* poses, int workers, bool unique);
int batch_cpu_solver_tft(cv::Point3f const* p3d_1, cv::Point2f const* p2d_2, cv::Point2f const* p2d_3, int point_count, cv::Mat const& K, int poses_to_sample, float* poses, float* next_pool, int workers, bool unique);
int batch_cpu_solver_rnp(cv::Point2f const* p2d, cv::Point3f const* p3d, int point_count, int sample_size, cv::Mat const& K, int solver, bool direction, float r0, int max_pow, int max_iterations, int poses_to_sample, float* poses, int workers, bool unique);






int batch_solve_ap3p_cpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
//int batch_solve_lambdatwist_cpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
int batch_solve_ap3p_gpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
int batch_solve_lambdatwist_gpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool);
