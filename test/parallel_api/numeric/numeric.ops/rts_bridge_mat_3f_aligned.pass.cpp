// -*- C++ -*-
//===-- rts_bridge_mat_3f_aligned.pass.cpp ----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 3f (Matrix2x2): Same as bridge 4b but with alignas(16) on the
// Matrix2x2 type. If this passes while 4b fails, alignment is the root cause.

#define _ONEDPL_REDUCE_THEN_SCAN_DEBUG 0

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <oneapi/dpl/pstl/utils.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>

namespace rts = oneapi::dpl::__par_backend_hetero;

// Aligned version of Matrix2x2
template <typename T>
struct alignas(16) AlignedMatrix2x2
{
    T a00, a01, a10, a11;
    AlignedMatrix2x2() : a00(1), a01(0), a10(0), a11(1) {}
    AlignedMatrix2x2(T x, T y) : a00(0), a01(x), a10(x), a11(y) {}
};

template <typename T>
bool
operator==(const AlignedMatrix2x2<T>& left, const AlignedMatrix2x2<T>& right)
{
    return left.a00 == right.a00 && left.a01 == right.a01 && left.a10 == right.a10 && left.a11 == right.a11;
}

template <typename T>
struct multiply_aligned_matrix
{
    AlignedMatrix2x2<T>
    operator()(const AlignedMatrix2x2<T>& left, const AlignedMatrix2x2<T>& right) const
    {
        AlignedMatrix2x2<T> result;
        result.a00 = left.a00 * right.a00 + left.a01 * right.a10;
        result.a01 = left.a00 * right.a01 + left.a01 * right.a11;
        result.a10 = left.a10 * right.a00 + left.a11 * right.a10;
        result.a11 = left.a10 * right.a01 + left.a11 * right.a11;
        return result;
    }
};

template <typename T, sycl::access::mode M = sycl::access::mode::read_write>
using all_view = oneapi::dpl::__ranges::all_view<T, M>;

// Kernel name tag
struct Bridge3fMatKernel;

int run_test() {
    using T = AlignedMatrix2x2<std::int32_t>;
    using _UnaryOp = oneapi::dpl::identity;
    using _BinaryOp = multiply_aligned_matrix<std::int32_t>;
    using _InitType = oneapi::dpl::unseq_backend::__no_init_value<T>;
    using _GenInput = rts::__gen_transform_input<_UnaryOp, T>;
    using _ScanInputTransform = oneapi::dpl::identity;
    using _WriteOp = rts::__simple_write_to_id;

    constexpr std::size_t N = 20000;

    std::printf("[bridge3f_mat] sizeof(AlignedMatrix2x2<int32_t>) = %zu\n", sizeof(T));
    std::printf("[bridge3f_mat] alignof(AlignedMatrix2x2<int32_t>) = %zu\n", alignof(T));
    std::printf("[bridge3f_mat] is_trivially_copyable = %d\n", (int)std::is_trivially_copyable_v<T>);

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = T(k % 7 + 1, k % 7 + 2);

    // Manual inclusive scan with matrix multiply
    _BinaryOp mat_op{};
    std::vector<T> h_expected(N);
    h_expected[0] = h_input[0];
    for (std::size_t i = 1; i < N; i++)
        h_expected[i] = mat_op(h_expected[i - 1], h_input[i]);

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::printf("[bridge3f_mat] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

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

    std::printf("[bridge3f_mat] Calling __parallel_transform_reduce_then_scan directly\n");

    auto future = rts::__parallel_transform_reduce_then_scan<sizeof(T), Bridge3fMatKernel>(
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
        if (!(h_out[i] == h_expected[i])) {
            if (errors < 20)
                std::printf("[bridge3f_mat] MISMATCH [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                            i, h_out[i].a00, h_out[i].a01, h_out[i].a10, h_out[i].a11,
                            h_expected[i].a00, h_expected[i].a01, h_expected[i].a10, h_expected[i].a11);
            errors++;
        }
    }
    std::printf("[bridge3f_mat] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

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
