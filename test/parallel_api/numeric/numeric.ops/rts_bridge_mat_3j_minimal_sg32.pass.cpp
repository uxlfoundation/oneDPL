// -*- C++ -*-
//===-- rts_bridge_mat_3j_minimal_sg32.pass.cpp ---------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Minimal reproducer: reqd_sub_group_size(32) + Matrix2x2 passthrough only.
// No sub-group ops, no SLM. If this segfaults on Windows CPU release,
// the JIT miscompiles the attribute + struct type combination.

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

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    std::printf("[bridge3j] Device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    std::vector<T> h_input(N);
    for (std::size_t i = 0; i < N; i++)
        h_input[i] = T(static_cast<std::int32_t>(i % 7 + 1), static_cast<std::int32_t>(i % 5 + 1));

    sycl::buffer<T, 1> buf_in(h_input.data(), sycl::range<1>(N));
    sycl::buffer<T, 1> buf_out{sycl::range<1>(N)};

    q.submit([&](sycl::handler& cgh) {
        auto in = buf_in.get_access<sycl::access::mode::read>(cgh);
        auto out = buf_out.get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class Bridge3jKernel>(
            sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(32)),
            [=](sycl::nd_item<1> ndi) [[sycl::reqd_sub_group_size(32)]] {
                std::size_t gid = ndi.get_global_linear_id();
                out[gid] = in[gid];
            });
    }).wait();

    std::vector<T> h_out(N);
    {
        auto acc = buf_out.get_host_access();
        for (std::size_t i = 0; i < N; i++)
            h_out[i] = acc[i];
    }

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (!(h_out[i] == h_input[i])) {
            if (errors < 10)
                std::printf("[bridge3j] MISMATCH [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                            i, h_out[i].a00, h_out[i].a01, h_out[i].a10, h_out[i].a11,
                            h_input[i].a00, h_input[i].a01, h_input[i].a10, h_input[i].a11);
            errors++;
        }
    }
    std::printf("[bridge3j] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);
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
