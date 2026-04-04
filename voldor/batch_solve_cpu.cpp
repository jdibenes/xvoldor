
#include <vector>
#include <thread>
#include <memory>
#include <iostream>
#include "batch_solve_cpu.h"

// OK
int batch_solve(int jobs, int workers, batch_callback f, 
    
    void* inputs, int point_count, int sample_size, void* output, int solution_size_bytes)
{
    if (jobs == 0) { return 0; }

    int batch = jobs / workers;
    int spill = jobs % workers;
    int start = 0;

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
        registry.push_back({ inputs, output, rng.get(), i, start, end, point_count, sample_size, 0 });
        start = end;
    }

    for (auto& tjd : registry) { threads.push_back(std::thread(f, std::ref(tjd))); }
    for (auto& wtp : threads) { wtp.join(); }

    int valid = registry[0].valid;
    std::cout << "VVVV: " << valid << " | ";
    for (int i = 1; i < registry.size(); ++i)
    {
        uint8_t* blk = static_cast<uint8_t*>(output);
        uint8_t* src = blk + (registry[i].start * solution_size_bytes);
        uint8_t* dst = blk + (valid * solution_size_bytes);
        memmove(dst, src, registry[i].valid * solution_size_bytes);
        valid += registry[i].valid;
        std::cout << registry[i].valid << " | ";
    }
    std::cout << std::endl;

    


    return valid;    
}


