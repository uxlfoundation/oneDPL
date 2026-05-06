// -*- C++ -*-
//===-- rts_bridge_mat_3k_slm_roundtrip.pass.cpp --------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Minimal reproducer: reqd_sub_group_size(32) + Matrix2x2 + SLM round-trip.
// Each work-item writes its Matrix2x2 to local memory, barrier, then reads
// from a neighbor (shift-by-1 via SLM). If 3j passes but this fails,
// the JIT miscompiles SLM access under forced 32-wide vectorization.

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <vector>

using TestUtils::Matrix2x2;

int run_test() {
    using T = Matrix2x2<std::int32_t>;
    constexpr std::size_t N = 256;
    constexpr std::size_t WG_SIZE = 32;

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    std::printf("[bridge3k] Device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    std::vector<T> h_input(N);
    for (std::size_t i = 0; i < N; i++)
        h_input[i] = T(static_cast<std::int32_t>(i % 7 + 1), static_cast<std::int32_t>(i % 5 + 1));

    sycl::buffer<T, 1> buf_in(h_input.data(), sycl::range<1>(N));
    sycl::buffer<T, 1> buf_out{sycl::range<1>(N)};

    q.submit([&](sycl::handler& cgh) {
        auto in = buf_in.get_access<sycl::access::mode::read>(cgh);
        auto out = buf_out.get_access<sycl::access::mode::read_write>(cgh);
        sycl::local_accessor<T, 1> slm(sycl::range<1>(WG_SIZE), cgh);
        cgh.parallel_for<class Bridge3kKernel>(
            sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> ndi) [[sycl::reqd_sub_group_size(32)]] {
                std::size_t gid = ndi.get_global_linear_id();
                std::size_t lid = ndi.get_local_linear_id();

                // Write to SLM
                slm[lid] = in[gid];

                sycl::group_barrier(ndi.get_group());

                // Read from neighbor (shift right by 1, wrap around)
                std::size_t src = (lid == 0) ? lid : lid - 1;
                out[gid] = slm[src];
            });
    }).wait();

    std::vector<T> h_out(N);
    {
        auto acc = buf_out.get_host_access();
        for (std::size_t i = 0; i < N; i++)
            h_out[i] = acc[i];
    }

    int errors = 0;
    for (std::size_t wg_start = 0; wg_start < N; wg_start += WG_SIZE) {
        // Lane 0 reads from itself (src=0)
        if (!(h_out[wg_start] == h_input[wg_start])) {
            if (errors < 10)
                std::printf("[bridge3k] MISMATCH [%zu]\n", wg_start);
            errors++;
        }
        // Other lanes read from lane-1
        for (std::size_t i = 1; i < WG_SIZE && (wg_start + i) < N; i++) {
            T expected = h_input[wg_start + i - 1];
            if (!(h_out[wg_start + i] == expected)) {
                if (errors < 10)
                    std::printf("[bridge3k] MISMATCH [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                                wg_start + i,
                                h_out[wg_start + i].a00, h_out[wg_start + i].a01,
                                h_out[wg_start + i].a10, h_out[wg_start + i].a11,
                                expected.a00, expected.a01, expected.a10, expected.a11);
                errors++;
            }
        }
    }
    std::printf("[bridge3k] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);
    return errors ? 1 : 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    int result = run_test();
    if (result != 0) return result;
#endif
    return TestUtils::done();
}
