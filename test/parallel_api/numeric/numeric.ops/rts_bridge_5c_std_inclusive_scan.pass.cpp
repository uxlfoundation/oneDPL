// -*- C++ -*-
//===-- rts_bridge_5c_std_inclusive_scan.pass.cpp ---------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 5c: Calls std::inclusive_scan directly with a device_policy and
// std::vector iterators, WITHOUT the invoke_on_all_policies test
// infrastructure. This exercises the full glue layer: type repacking via
// __repacked_tuple_t, __select_backend dispatch, __pattern_transform_scan.
// The only difference from scan_int32_no_single_group is the absence of the
// invoke_on_all_policies / iterator_invoker / Sequence test framework.

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
    // Use serial std::inclusive_scan for expected values
    std::inclusive_scan(h_input.begin(), h_input.end(), h_expected.begin());

    std::vector<T> h_out(N, T{-666});

    auto policy = oneapi::dpl::execution::make_device_policy(sycl::queue{sycl::default_selector_v});
    auto dev = policy.queue().get_device();
    std::printf("[bridge5c] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    std::printf("[bridge5c] Calling std::inclusive_scan with device_policy (no test infrastructure)\n");

    // Full std:: API path, but without invoke_on_all_policies / Sequence / iterator_invoker
    auto result = std::inclusive_scan(policy, h_input.begin(), h_input.end(), h_out.begin());

    // Verify the returned iterator
    if (result != h_out.begin() + N) {
        std::printf("[bridge5c] ERROR: returned iterator is wrong\n");
        return 1;
    }

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (h_out[i] != h_expected[i]) {
            if (errors < 20)
                std::printf("[bridge5c] MISMATCH [%zu]: got %d expected %d\n", i, h_out[i], h_expected[i]);
            errors++;
        }
    }
    std::printf("[bridge5c] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

    // Also test exclusive_scan
    std::vector<T> h_out_ex(N, T{-666});
    std::vector<T> h_expected_ex(N);
    std::exclusive_scan(h_input.begin(), h_input.end(), h_expected_ex.begin(), T{0});

    std::printf("[bridge5c] Calling std::exclusive_scan with device_policy (no test infrastructure)\n");

    auto result_ex = std::exclusive_scan(policy, h_input.begin(), h_input.end(), h_out_ex.begin(), T{0});

    if (result_ex != h_out_ex.begin() + N) {
        std::printf("[bridge5c] ERROR: exclusive_scan returned iterator is wrong\n");
        return 1;
    }

    int errors_ex = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (h_out_ex[i] != h_expected_ex[i]) {
            if (errors_ex < 20)
                std::printf("[bridge5c] MISMATCH ex [%zu]: got %d expected %d\n", i, h_out_ex[i], h_expected_ex[i]);
            errors_ex++;
        }
    }
    std::printf("[bridge5c] exclusive: %s: %d errors out of %zu\n", errors_ex ? "FAIL" : "PASS", errors_ex, N);

    return (errors || errors_ex) ? 1 : 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    int result = run_test();
    if (result != 0) return result;
#endif
    return TestUtils::done();
}
