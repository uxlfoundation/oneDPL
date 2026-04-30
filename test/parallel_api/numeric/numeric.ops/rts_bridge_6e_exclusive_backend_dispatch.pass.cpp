// -*- C++ -*-
//===-- rts_bridge_6e_exclusive_backend_dispatch.pass.cpp -------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 6e: Calls __parallel_transform_scan (SYCL backend dispatch) with
// a device_policy for exclusive scan. Analog of bridge 5a but exclusive.
// Adds over 6c: policy-derived kernel name, single-group check dispatch.
// Gap between 6e and 6d: __get_sycl_range iterator-to-buffer conversion.

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
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>

namespace rts = oneapi::dpl::__par_backend_hetero;

template <typename T, sycl::access::mode M = sycl::access::mode::read_write>
using all_view = oneapi::dpl::__ranges::all_view<T, M>;

int run_test() {
    using T = std::int32_t;
    using _UnaryOp = oneapi::dpl::identity;
    using _BinaryOp = std::plus<T>;
    using _InitType = oneapi::dpl::unseq_backend::__init_value<T>;

    constexpr std::size_t N = 20000;

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = static_cast<T>((k % 991 + 1) ^ (k % 997 + 2));
    std::vector<T> h_expected(N);
    std::exclusive_scan(h_input.begin(), h_input.end(), h_expected.begin(), T{0});

    auto policy = oneapi::dpl::execution::make_device_policy(sycl::queue{sycl::default_selector_v});
    sycl::queue q = policy.queue();
    auto dev = q.get_device();
    std::printf("[bridge6e] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    sycl::buffer<T, 1> buf_in(h_input.data(), sycl::range<1>(N));
    sycl::buffer<T, 1> buf_out{sycl::range<1>(N)};

    all_view<T, sycl::access::mode::read> in_view(buf_in, 0, N);
    all_view<T, sycl::access::mode::read_write> out_view(buf_out, 0, N);

    _UnaryOp unary_op{};
    _BinaryOp binary_op{};
    _InitType init{T{0}};

    std::printf("[bridge6e] Calling __parallel_transform_scan via device_policy (exclusive)\n");

    rts::__parallel_transform_scan(
        oneapi::dpl::__internal::__device_backend_tag{},
        std::forward<decltype(policy)>(policy),
        in_view, out_view, N,
        unary_op, init, binary_op,
        std::false_type{} /*exclusive*/)
        .__checked_deferrable_wait();

    std::vector<T> h_out(N);
    {
        auto acc = buf_out.get_host_access();
        for (std::size_t i = 0; i < N; i++)
            h_out[i] = acc[i];
    }

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (h_out[i] != h_expected[i]) {
            if (errors < 20)
                std::printf("[bridge6e] MISMATCH [%zu]: got %d expected %d\n", i, h_out[i], h_expected[i]);
            errors++;
        }
    }
    std::printf("[bridge6e] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

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
