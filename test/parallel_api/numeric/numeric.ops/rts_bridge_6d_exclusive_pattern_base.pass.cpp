// -*- C++ -*-
//===-- rts_bridge_6d_exclusive_pattern_base.pass.cpp -----------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 6d: Calls __pattern_transform_scan_base with host iterators
// for exclusive scan. Based on rts_bridge_5b_pattern_scan_base.pass.cpp.
// Key changes from bridge 5b:
//   - __init_value<_RepackedType> instead of __no_init_value<_RepackedType>
//   - std::false_type{} for inclusive instead of std::true_type{}
//   - Expected output: std::exclusive_scan(..., T{0})

#define _ONEDPL_REDUCE_THEN_SCAN_DEBUG 0

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/pstl/hetero/numeric_impl_hetero.h>

int run_test() {
    using T = std::int32_t;
    using _UnaryOp = oneapi::dpl::identity;
    using _BinaryOp = std::plus<T>;

    constexpr std::size_t N = 20000;

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = static_cast<T>((k % 991 + 1) ^ (k % 997 + 2));
    std::vector<T> h_expected(N);
    std::exclusive_scan(h_input.begin(), h_input.end(), h_expected.begin(), T{0});

    std::vector<T> h_out(N, T{-666});

    auto policy = oneapi::dpl::execution::make_device_policy(sycl::queue{sycl::default_selector_v});
    sycl::queue q = policy.queue();
    auto dev = q.get_device();
    std::printf("[bridge6d] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    _UnaryOp unary_op{};
    _BinaryOp binary_op{};

    // EXCLUSIVE: Use __init_value with init = 0
    using _RepackedType = oneapi::dpl::__par_backend_hetero::__repacked_tuple_t<T>;
    using _InitType = oneapi::dpl::unseq_backend::__init_value<_RepackedType>;
    _InitType init{_RepackedType{0}};

    std::printf("[bridge6d] Calling __pattern_transform_scan_base with host iterators (exclusive)\n");

    // This is the hetero pattern layer (numeric_impl_hetero.h)
    // It converts host iterators to sycl ranges via __get_sycl_range,
    // then calls __parallel_transform_scan
    auto __tag = oneapi::dpl::__internal::__hetero_tag<oneapi::dpl::__internal::__device_backend_tag>{};
    oneapi::dpl::__internal::__pattern_transform_scan_base(
        __tag,
        std::forward<decltype(policy)>(policy),
        h_input.begin(), h_input.end(),
        h_out.begin(),
        unary_op, init, binary_op,
        std::false_type{} /*exclusive*/);

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (h_out[i] != h_expected[i]) {
            if (errors < 20)
                std::printf("[bridge6d] MISMATCH [%zu]: got %d expected %d\n", i, h_out[i], h_expected[i]);
            errors++;
        }
    }
    std::printf("[bridge6d] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

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
