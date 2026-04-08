// Reproducer for register spill warnings in oneDPL sort_by_key
// with subgroup radix sort on PVC/BMG GPUs.
//
// Build:
//    icpx -fsycl -std=c++17 -O2 reproducer_sort_spill.cpp -Wall -Wextra -Werror -Wimplicit-fallthrough -Wformat -Wformat-security -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc" -ftarget-register-alloc-mode=pvc:default -fsycl-force-target=spir64 -Wno-sycl-target -o reproducer_sort_spill -I./include
//
// Expected: build warnings about register allocation and spilling, e.g.:
//   warning: kernel ... compiled SIMD16 allocated 128 regs and spilled around 106
//
// To target a specific device:
//   ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./reproducer_sort_spill

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

bool
test_sort_by_key(sycl::queue& q, std::size_t n)
{
    auto* keys = sycl::malloc_device<std::int32_t>(n, q);
    auto* vals = sycl::malloc_device<std::int32_t>(n, q);

    // Initialize with descending keys, identity permutation values
    {
        std::vector<std::int32_t> h_keys(n);
        std::vector<std::int32_t> h_vals(n);
        std::iota(h_keys.rbegin(), h_keys.rend(), 0);  // descending
        std::iota(h_vals.begin(), h_vals.end(), 0);     // 0,1,2,...

        q.memcpy(keys, h_keys.data(), n * sizeof(std::int32_t));
        q.memcpy(vals, h_vals.data(), n * sizeof(std::int32_t));
        q.wait();
    }

    auto policy = oneapi::dpl::execution::make_device_policy(q);
    oneapi::dpl::sort_by_key(policy, keys, keys + n, vals, std::less<void>());

    // Quick correctness check
    std::vector<std::int32_t> h_keys(n), h_vals(n);
    q.memcpy(h_keys.data(), keys, n * sizeof(std::int32_t));
    q.memcpy(h_vals.data(), vals, n * sizeof(std::int32_t));
    q.wait();

    bool ok = true;
    for (std::size_t i = 1; i < n && ok; ++i)
        ok = (h_keys[i - 1] <= h_keys[i]);

    std::cout << "n = " << n << ", sorted correctly: " << (ok ? "yes" : "NO") << "\n";

    sycl::free(keys, q);
    sycl::free(vals, q);
    return ok;
}

int
main()
{
    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Common sizes from HPCG benchmark
    constexpr std::size_t sizes[] = {1048576, 1024000, 2097152, 2048000, 4194304};

    bool all_ok = true;
    for (auto n : sizes)
        all_ok &= test_sort_by_key(q, n);

    return all_ok ? 0 : 1;
}
