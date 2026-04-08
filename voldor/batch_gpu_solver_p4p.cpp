
#include <memory>
#include "../gpu-kernels/gpu_kernels.h"
#include "batch_cpu_solver.h"
#include "helpers_opencv.h"

int batch_gpu_solver_p4p(cv::Point3f const* p3d_1, cv::Point2f const* p2k_2, int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses)
{
    if (point_count < 4) { return 0; }

    std::unique_ptr<cv::Vec3f[]> r_s = std::make_unique<cv::Vec3f[]>(poses_to_sample);
    std::unique_ptr<cv::Vec3f[]> t_s = std::make_unique<cv::Vec3f[]>(poses_to_sample);

    switch (solver)
    {
    case 0:  solve_batch_p3p_ap3p_gpu(       reinterpret_cast<float const*>(p3d_1), reinterpret_cast<float const*>(p2k_2), reinterpret_cast<float*>(r_s.get()), reinterpret_cast<float*>(t_s.get()), reinterpret_cast<float*>(K.data), point_count, poses_to_sample); break;
    case 1:  solve_batch_p3p_lambdatwist_gpu(reinterpret_cast<float const*>(p3d_1), reinterpret_cast<float const*>(p2k_2), reinterpret_cast<float*>(r_s.get()), reinterpret_cast<float*>(t_s.get()), reinterpret_cast<float*>(K.data), point_count, poses_to_sample); break;
    default: return 0;
    }

    job_descriptor jd{ nullptr, nullptr, nullptr, 0, 0, poses_to_sample, point_count, 4, 0 };

    for (int i = jd.start; i < jd.end; ++i)
    {
    if (!is_valid_solution_6(r_s[i].val, t_s[i].val)) { continue; }
    put_solution_6(jd, poses, r_s[i].val, t_s[i].val);
    jd.valid++;
    }

    return jd.valid;
}





/*
#include <opencv2/calib3d.hpp>
#include "../gpu-kernels/gpu_kernels.h"

int batch_solve_lambdatwist_gpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool)
{
    int n_points = (int)pts2.size();
    int poses_pool_used = 0;

    float* ret_Rs = new float[poses_to_sample * 3];
    float* ret_ts = new float[poses_to_sample * 3];

    solve_batch_p3p_lambdatwist_gpu((float*)pts3.data(), (float*)pts2.data(), ret_Rs, ret_ts, (float*)K.data, n_points, poses_to_sample);

    for (int i = 0; i < poses_to_sample; i++)
    {
        float r_sum = ret_Rs[i * 3 + 0] + ret_Rs[i * 3 + 1] + ret_Rs[i * 3 + 2];
        float t_sum = ret_ts[i * 3 + 0] + ret_ts[i * 3 + 1] + ret_ts[i * 3 + 2];

        float x_sum = r_sum + t_sum;

        if (!isfinite(x_sum)) { continue; }

        poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = cv::Vec3f(ret_Rs[i * 3 + 0], ret_Rs[i * 3 + 1], ret_Rs[i * 3 + 2]);
        poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = cv::Vec3f(ret_ts[i * 3 + 0], ret_ts[i * 3 + 1], ret_ts[i * 3 + 2]);
        poses_pool_used++;
    }

    delete[] ret_Rs;
    delete[] ret_ts;

    return poses_pool_used;
}


#include <opencv2/calib3d.hpp>
#include "../gpu-kernels/gpu_kernels.h"

int batch_solve_ap3p_gpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool)
{
    int n_points = (int)pts2.size();
    int poses_pool_used = 0;

    float* ret_Rs = new float[poses_to_sample * 3];
    float* ret_ts = new float[poses_to_sample * 3];

    solve_batch_p3p_ap3p_gpu((float*)pts3.data(), (float*)pts2.data(), ret_Rs, ret_ts, (float*)K.data, n_points, poses_to_sample);

    for (int i = 0; i < poses_to_sample; i++)
    {
        float r_sum = ret_Rs[i * 3 + 0] + ret_Rs[i * 3 + 1] + ret_Rs[i * 3 + 2];
        float t_sum = ret_ts[i * 3 + 0] + ret_ts[i * 3 + 1] + ret_ts[i * 3 + 2];

        float x_sum = r_sum + t_sum;

        if (!isfinite(x_sum)) { continue; }

        poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = cv::Vec3f(ret_Rs[i * 3 + 0], ret_Rs[i * 3 + 1], ret_Rs[i * 3 + 2]);
        poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = cv::Vec3f(ret_ts[i * 3 + 0], ret_ts[i * 3 + 1], ret_ts[i * 3 + 2]);
        poses_pool_used++;
    }

    delete[] ret_Rs;
    delete[] ret_ts;

    return poses_pool_used;
}
*/