// -*- C++ -*-
//===-- rts_bridge_mat_5a_parallel_transform_scan.pass.cpp ----------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test mat_5a: Matrix2x2 version of bridge 5a. Calls
// __parallel_transform_scan with a device_policy, buffer-backed all_view
// ranges, using non-commutative matrix multiplication.

#define _ONEDPL_REDUCE_THEN_SCAN_DEBUG 0

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>

namespace rts = oneapi::dpl::__par_backend_hetero;

template <typename T, sycl::access::mode M = sycl::access::mode::read_write>
using all_view = oneapi::dpl::__ranges::all_view<T, M>;

int run_test() {
    using T = Matrix2x2<std::int32_t>;
    using _UnaryOp = oneapi::dpl::identity;
    using _BinaryOp = multiply_matrix<std::int32_t>;
    using _InitType = oneapi::dpl::unseq_backend::__no_init_value<T>;

    constexpr std::size_t N = 10000;

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = T(k % 7 + 1, k % 7 + 2);
    std::vector<T> h_expected(N);
    _BinaryOp op{};
    h_expected[0] = h_input[0];
    for (std::size_t i = 1; i < N; i++)
        h_expected[i] = op(h_expected[i - 1], h_input[i]);

    auto policy = oneapi::dpl::execution::make_device_policy(sycl::queue{sycl::default_selector_v});
    auto dev = policy.queue().get_device();
    std::printf("[bridge_mat_5a] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    sycl::buffer<T, 1> buf_in(h_input.data(), sycl::range<1>(N));
    sycl::buffer<T, 1> buf_out{sycl::range<1>(N)};

    all_view<T, sycl::access::mode::read> in_view(buf_in, 0, N);
    all_view<T, sycl::access::mode::read_write> out_view(buf_out, 0, N);

    _UnaryOp unary_op{};
    _BinaryOp binary_op{};
    _InitType init{};

    std::printf("[bridge_mat_5a] Calling __parallel_transform_scan via device_policy\n");

    rts::__parallel_transform_scan(
        oneapi::dpl::__internal::__device_backend_tag{},
        std::forward<decltype(policy)>(policy),
        in_view, out_view, N,
        unary_op, init, binary_op,
        std::true_type{} /*inclusive*/)
        .__checked_deferrable_wait();

    std::vector<T> h_out(N);
    {
        auto acc = buf_out.get_host_access();
        for (std::size_t i = 0; i < N; i++)
            h_out[i] = acc[i];
    }

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (!(h_out[i] == h_expected[i])) {
            if (errors < 10)
                std::printf("[bridge_mat_5a] MISMATCH [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                    i, h_out[i].a00, h_out[i].a01, h_out[i].a10, h_out[i].a11,
                    h_expected[i].a00, h_expected[i].a01, h_expected[i].a10, h_expected[i].a11);
            errors++;
        }
    }
    std::printf("[bridge_mat_5a] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);
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
