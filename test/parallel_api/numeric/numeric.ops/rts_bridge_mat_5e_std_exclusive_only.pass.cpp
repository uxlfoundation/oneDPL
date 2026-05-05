// -*- C++ -*-
//===-- rts_bridge_mat_5e_std_exclusive_only.pass.cpp ---------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test mat_5e: Matrix2x2 version of bridge 5e. Calls std::exclusive_scan
// directly with device_policy using non-commutative matrix multiplication.

#define _ONEDPL_REDUCE_THEN_SCAN_DEBUG 0

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <cstdint>
#include <cstdio>
#include <vector>

int run_test() {
    using T = Matrix2x2<std::int32_t>;
    using _BinaryOp = multiply_matrix<std::int32_t>;

    constexpr std::size_t N = 20000;

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = T(k % 7 + 1, k % 7 + 2);
    std::vector<T> h_expected(N);
    _BinaryOp op{};
    h_expected[0] = T();
    for (std::size_t i = 1; i < N; i++)
        h_expected[i] = op(h_expected[i - 1], h_input[i - 1]);

    std::vector<T> h_out(N, T(-666, 666));

    auto policy = oneapi::dpl::execution::make_device_policy(sycl::queue{sycl::default_selector_v});
    auto dev = policy.queue().get_device();
    std::printf("[bridge_mat_5e] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("[bridge_mat_5e] Calling std::exclusive_scan ONLY with device_policy\n");

    auto result = std::exclusive_scan(policy, h_input.begin(), h_input.end(), h_out.begin(), T(), op);

    if (result != h_out.begin() + N) {
        std::printf("[bridge_mat_5e] ERROR: returned iterator is wrong\n");
        return 1;
    }

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (!(h_out[i] == h_expected[i])) {
            if (errors < 10)
                std::printf("[bridge_mat_5e] MISMATCH [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                    i, h_out[i].a00, h_out[i].a01, h_out[i].a10, h_out[i].a11,
                    h_expected[i].a00, h_expected[i].a01, h_expected[i].a10, h_expected[i].a11);
            errors++;
        }
    }
    std::printf("[bridge_mat_5e] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);
    return errors ? 1 : 0;
}

#endif

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    int result = run_test();
    if (result != 0) return result;
#endif
    return TestUtils::done();
}
