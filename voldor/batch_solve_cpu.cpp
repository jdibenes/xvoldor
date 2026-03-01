
#include <vector>
#include <thread>
#include <memory>
#include "batch_solve_cpu.h"

// OK
int batch_solve(int jobs, int workers, batch_callback f, void* data, int point_count, int sample_size)
{
    int batch = jobs / workers;
    int spill = jobs % workers;
    int start = 0;

    std::atomic<int> valid = 0;

    std::vector<job_descriptor> registry;    
    std::vector<std::thread> threads;

    // for determinism
    std::unique_ptr<int[]> rng = std::make_unique<int[]>(sample_size * jobs);
    for (int i = 0; i < jobs; ++i) { for (int p = 0; p < sample_size; ++p) { rng[(sample_size * i) + p] = static_cast<int>((static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) * (point_count - 1)); } }

    for (int i = 0; i < workers; ++i)
    {
        int end = start + batch;
        if (spill > 0)
        {
            end++;
            spill--;
        }
        if (start >= end) { break; }
        registry.push_back({ rng.get(), i, start, end, point_count, sample_size });
        start = end;
    }

    for (auto& tjd : registry) { threads.push_back(std::thread(f, data, std::ref(tjd), std::ref(valid))); }
    for (auto& wtp : threads) { wtp.join(); }

    return valid;    
}
