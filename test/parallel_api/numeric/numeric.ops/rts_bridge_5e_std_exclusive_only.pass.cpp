// -*- C++ -*-
//===-- rts_bridge_5e_std_exclusive_only.pass.cpp ---------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 5e: Same as 5c but ONLY exclusive_scan (no inclusive_scan).
// Disambiguates whether 5c's crash is from exclusive_scan alone, or from
// having both scan variants instantiated in the same binary.

#define _ONEDPL_REDUCE_THEN_SCAN_DEBUG 0

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

int run_test() {
    using T = std::int32_t;

    constexpr std::size_t N = 20000;

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = static_cast<T>((k % 991 + 1) ^ (k % 997 + 2));
    std::vector<T> h_expected(N);
    std::exclusive_scan(h_input.begin(), h_input.end(), h_expected.begin(), T{0});

    std::vector<T> h_out(N, T{-666});

    auto policy = oneapi::dpl::execution::make_device_policy(sycl::queue{sycl::default_selector_v});
    auto dev = policy.queue().get_device();
    std::printf("[bridge5e] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    std::printf("[bridge5e] Calling std::exclusive_scan ONLY with device_policy\n");

    auto result = std::exclusive_scan(policy, h_input.begin(), h_input.end(), h_out.begin(), T{0});

    if (result != h_out.begin() + N) {
        std::printf("[bridge5e] ERROR: returned iterator is wrong\n");
        return 1;
    }

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (h_out[i] != h_expected[i]) {
            if (errors < 20)
                std::printf("[bridge5e] MISMATCH [%zu]: got %d expected %d\n", i, h_out[i], h_expected[i]);
            errors++;
        }
    }
    std::printf("[bridge5e] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

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
