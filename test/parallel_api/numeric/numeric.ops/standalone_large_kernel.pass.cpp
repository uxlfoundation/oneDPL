// -*- C++ -*-
//===-- standalone_large_kernel.pass.cpp ----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Standalone SYCL test: Kernel with a deliberately large stack frame
// (many local variables, arrays) to test whether /RTC1 debug builds
// on Windows CPU trigger stack buffer overrun from stack pressure alone,
// independent of oneDPL logic. This mimics the approximate stack footprint
// of the RTS reduce/scan kernel bodies.

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <vector>

// Approximate the stack frame of an RTS kernel:
// - __max_inputs_per_item=128 for int32 (512/4): 128 lazy_ctor_storage<int32> = 512 bytes
// - sub_group_carry (lazy storage): ~4 bytes
// - sub_group_params: ~20 bytes
// - Various loop vars, indices: ~64 bytes
// - Multiple function call frames for __scan_through_elements_helper etc
// Total per work-item: ~1-2 KB on stack (amplified by /RTC1 instrumentation)

struct big_stack_state
{
    std::int32_t data[128];
    std::int32_t carry;
    std::uint32_t params[8];
};

int
run_large_kernel()
{
    sycl::queue q;
    constexpr std::uint32_t wg_size = 256;
    constexpr std::uint32_t num_wg = 8;
    constexpr std::uint32_t global_size = wg_size * num_wg;

    std::vector<std::int32_t> output(global_size, 0);
    {
        sycl::buffer<std::int32_t> out_buf(output.data(), sycl::range<1>(global_size));

        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<std::int32_t, 1> slm(sycl::range<1>(wg_size), cgh);
            auto out_acc = out_buf.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class large_kernel>(
                sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(wg_size)),
                [=](sycl::nd_item<1> ndi) {
                    std::uint32_t global_id = ndi.get_global_linear_id();
                    std::uint32_t local_id = ndi.get_local_linear_id();

                    // Large stack frame: local array + struct
                    big_stack_state state;
                    for (int i = 0; i < 128; ++i)
                        state.data[i] = static_cast<std::int32_t>(global_id + i);
                    state.carry = 0;
                    for (int i = 0; i < 8; ++i)
                        state.params[i] = local_id + i;

                    // Simulate scan-like reduction through local data
                    for (int i = 1; i < 128; ++i)
                        state.data[i] += state.data[i - 1];

                    state.carry = state.data[127];

                    // Use SLM
                    slm[local_id] = state.carry;
                    sycl::group_barrier(ndi.get_group());

                    // Read from neighbor
                    std::int32_t neighbor = slm[local_id > 0 ? local_id - 1 : 0];
                    state.carry += neighbor;

                    // Second large stack allocation to test deeper call stacks
                    std::int32_t accum[64];
                    for (int i = 0; i < 64; ++i)
                        accum[i] = state.data[i * 2] + state.carry;
                    for (int i = 1; i < 64; ++i)
                        accum[i] += accum[i - 1];

                    out_acc[global_id] = accum[63] + state.params[0];
                });
        }).wait();
    }

    bool ok = false;
    for (std::uint32_t i = 0; i < global_size; ++i)
    {
        if (output[i] != 0)
        {
            ok = true;
            break;
        }
    }
    if (!ok)
    {
        std::printf("FAIL: standalone_large_kernel output is all zeros\n");
        return 1;
    }
    return 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    int result = run_large_kernel();
    if (result != 0)
        return result;
#endif
    return TestUtils::done();
}
