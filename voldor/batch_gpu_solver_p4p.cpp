
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
