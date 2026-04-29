// -*- C++ -*-
//===-- standalone_slm_stress.pass.cpp ------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Standalone SYCL test: Allocates SLM matching the reduce_then_scan pattern
// (256 elements * sizeof(int32) = 1KB for comm_slm, plus sub_group_partials)
// and performs reads/writes. Tests whether SLM emulation on CPU under /RTC1
// triggers stack buffer overrun by itself, without oneDPL.

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <vector>

int
run_slm_stress()
{
    sycl::queue q;
    constexpr std::uint32_t wg_size = 256;
    constexpr std::uint32_t num_wg = 8;
    constexpr std::uint32_t global_size = wg_size * num_wg;
    constexpr std::uint32_t sg_size_assumed = 32;
    constexpr std::uint32_t num_sg_local = wg_size / sg_size_assumed;

    std::vector<std::int32_t> output(global_size, 0);
    {
        sycl::buffer<std::int32_t> out_buf(output.data(), sycl::range<1>(global_size));

        q.submit([&](sycl::handler& cgh) {
            // SLM matching RTS pattern: comm_slm (wg_size elements) + sub_group_partials (num_sg_local elements)
            sycl::local_accessor<std::int32_t, 1> comm_slm(sycl::range<1>(wg_size), cgh);
            sycl::local_accessor<std::int32_t, 1> sg_partials(sycl::range<1>(num_sg_local), cgh);
            auto out_acc = out_buf.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class slm_stress_kernel>(
                sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(wg_size)),
                [=](sycl::nd_item<1> ndi) {
                    auto sg = ndi.get_sub_group();
                    std::uint32_t local_id = ndi.get_local_linear_id();
                    std::uint32_t sg_id = sg.get_group_linear_id();
                    std::uint32_t global_id = ndi.get_global_linear_id();

                    // Write to comm_slm (mimics shift_group_right emulation)
                    comm_slm[local_id] = static_cast<std::int32_t>(global_id);
                    sycl::group_barrier(ndi.get_group());

                    // Read from neighbor (mimics shift by 1)
                    std::int32_t val = comm_slm[local_id > 0 ? local_id - 1 : local_id];

                    // Write sub-group partial (mimics reduce step writing to sg_partials)
                    if (sg.get_local_linear_id() == 0)
                        sg_partials[sg_id] = val;

                    sycl::group_barrier(ndi.get_group());

                    // Read back sub-group partial
                    if (sg_id == 0 && sg.get_local_linear_id() < num_sg_local)
                        val = sg_partials[sg.get_local_linear_id()];

                    out_acc[global_id] = val;
                });
        }).wait();
    }

    // Basic sanity: check output isn't all zeros
    bool ok = false;
    for (std::uint32_t i = 1; i < global_size; ++i)
    {
        if (output[i] != 0)
        {
            ok = true;
            break;
        }
    }
    if (!ok)
    {
        std::printf("FAIL: standalone_slm_stress output is all zeros\n");
        return 1;
    }
    return 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    int result = run_slm_stress();
    if (result != 0)
        return result;
#endif
    return TestUtils::done();
}
