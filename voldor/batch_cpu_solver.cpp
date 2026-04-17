
#include <vector>
#include <thread>
#include <memory>
#include <cmath>
#include "batch_cpu_solver.h"

// OK
static void sample(int n, int k, int* chosen, bool unique)
{
    int i = 0;
    while (i < k)
    {
    int s = static_cast<int>((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (n - 1));
    if ((n >= k) && unique)
    {
    int j;
    for (j = 0; (j < i) && (s != chosen[j]); ++j);
    if (j < i) { continue; }
    }
    chosen[i] = s;
    i++;
    }
}

// OK
std::vector<job_result> batch_solve(int jobs, int workers, batch_callback f, void* inputs, int point_count, int sample_size, bool unique, void* output)
{
    int batch = jobs / workers;
    int spill = jobs % workers;
    int start = 0;

    std::vector<job_descriptor> registry;
    std::vector<std::thread> threads;
    std::vector<job_result> results;

    std::unique_ptr<int[]> rng = std::make_unique<int[]>(sample_size * jobs);
    for (int i = 0; i < jobs; ++i) { sample(point_count, sample_size, &rng[sample_size * i], unique); }

    for (int i = 0; i < workers; ++i)
    {
    int end = start + batch;
    if (spill > 0)
    {
    end++;
    spill--;
    }
    if (start >= end) { break; }
    registry.push_back({ inputs, output, rng.get(), i, start, end, point_count, sample_size, 0 });
    start = end;
    }

    for (auto& tjd : registry) { threads.push_back(std::thread(f, std::ref(tjd))); }
    for (auto& wtp : threads) { wtp.join(); }
    for (auto& tjd : registry) { results.push_back({ tjd.id, tjd.start, tjd.end, tjd.valid }); }

    return results;
}

int batch_finalize(std::vector<job_result> const& jr, void* output, int solution_elements)
{
    uint8_t* base = static_cast<uint8_t*>(output);

    int solution_size_bytes = solution_elements * sizeof(float);
    int valid = 0;

    for (auto const& tjd : jr)
    {
    uint8_t* src = base + (solution_size_bytes * tjd.start);
    uint8_t* dst = base + (solution_size_bytes * valid);
    if (dst != src) { memmove(dst, src, solution_size_bytes * tjd.valid); }
    valid += tjd.valid;
    }

    return valid;
}

int const* get_sample_indices(job_descriptor const& jd, int index)
{
    return jd.rng + (index * jd.sample_size);
}

bool is_valid_solution_6(float const* r, float const* t)
{
    float r_sum = r[0] + r[1] + r[2];
    float t_sum = t[0] + t[1] + t[2];
    float x_sum = r_sum + t_sum;

    return std::isfinite(x_sum);
}

bool is_valid_solution_f(float f)
{
    return std::isfinite(f) && (f > 0);
}

void put_solution_6(job_descriptor& jd, float* base, float const* r, float const* t)
{
    float* out = &base[6 * (jd.start + jd.valid)];

    out[0] = r[0];
    out[1] = r[1];
    out[2] = r[2];
    out[3] = t[0];
    out[4] = t[1];
    out[5] = t[2];
}

void put_solution_f(job_descriptor& jd, float* base, float f)
{
    float* out = &base[1 * (jd.start + jd.valid)];

    out[0] = f;
}
