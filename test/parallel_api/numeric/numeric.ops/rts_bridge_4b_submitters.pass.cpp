// -*- C++ -*-
//===-- rts_bridge_4b_submitters.pass.cpp -----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 4b: Calls __parallel_transform_reduce_then_scan directly with
// buffer-backed all_view ranges, exercising the full submitter structs
// (__parallel_reduce_then_scan_reduce_submitter / scan_submitter) with their
// [=, *this] kernel captures. This is the last layer before full oneDPL.
// Tests whether the submitter struct capture triggers the crash.

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

#include <oneapi/dpl/pstl/utils.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>

namespace rts = oneapi::dpl::__par_backend_hetero;

template <typename T, sycl::access::mode M = sycl::access::mode::read_write>
using all_view = oneapi::dpl::__ranges::all_view<T, M>;

// Kernel name tag
struct Bridge4bKernel;

int run_test() {
    using T = std::int32_t;
    using _UnaryOp = oneapi::dpl::identity;
    using _BinaryOp = std::plus<T>;
    using _InitType = oneapi::dpl::unseq_backend::__no_init_value<T>;
    using _GenInput = rts::__gen_transform_input<_UnaryOp, T>;
    using _ScanInputTransform = oneapi::dpl::identity;
    using _WriteOp = rts::__simple_write_to_id;

    constexpr std::size_t N = 20000;

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = static_cast<T>((k % 991 + 1) ^ (k % 997 + 2));
    std::vector<T> h_expected(N);
    std::inclusive_scan(h_input.begin(), h_input.end(), h_expected.begin());

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::printf("[bridge4b] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    // Create buffer-backed all_view ranges
    sycl::buffer<T, 1> buf_in(h_input.data(), sycl::range<1>(N));
    sycl::buffer<T, 1> buf_out{sycl::range<1>(N)};

    all_view<T, sycl::access::mode::read> in_view(buf_in, 0, N);
    all_view<T, sycl::access::mode::read_write> out_view(buf_out, 0, N);

    _GenInput gen_input{_UnaryOp{}};
    _BinaryOp binary_op{};
    _ScanInputTransform scan_xform{};
    _WriteOp write_op{};
    _InitType init{};

    std::printf("[bridge4b] Calling __parallel_transform_reduce_then_scan directly\n");

    auto future = rts::__parallel_transform_reduce_then_scan<sizeof(T), Bridge4bKernel>(
        q, N, in_view, out_view,
        gen_input, binary_op, gen_input, scan_xform, write_op,
        init, /*_Inclusive=*/std::true_type{}, /*_IsUniquePattern=*/std::false_type{});

    future.__checked_deferrable_wait();

    // Read output back
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
                std::printf("[bridge4b] MISMATCH [%zu]: got %d expected %d\n", i, h_out[i], h_expected[i]);
            errors++;
        }
    }
    std::printf("[bridge4b] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

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
