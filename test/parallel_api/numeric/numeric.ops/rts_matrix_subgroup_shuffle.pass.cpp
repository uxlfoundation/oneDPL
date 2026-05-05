// -*- C++ -*-
//===-- rts_matrix_subgroup_shuffle.pass.cpp --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Reproducer: validates sycl::shift_group_right and sycl::group_broadcast on
// Matrix2x2<int32_t> (trivially-copyable 16-byte struct requiring multi-register
// shuffles). Tests the exact Hillis-Steele inclusive scan pattern used by
// oneDPL's reduce-then-scan algorithm.

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

template <typename T>
struct Matrix2x2
{
    T a00, a01, a10, a11;
    Matrix2x2() : a00(1), a01(0), a10(0), a11(1) {}
    Matrix2x2(T x, T y) : a00(0), a01(x), a10(x), a11(y) {}
};

template <typename T>
bool
operator==(const Matrix2x2<T>& l, const Matrix2x2<T>& r)
{
    return l.a00 == r.a00 && l.a01 == r.a01 && l.a10 == r.a10 && l.a11 == r.a11;
}

template <typename T>
struct multiply_matrix
{
    Matrix2x2<T>
    operator()(const Matrix2x2<T>& left, const Matrix2x2<T>& right) const
    {
        Matrix2x2<T> result;
        result.a00 = left.a00 * right.a00 + left.a01 * right.a10;
        result.a01 = left.a00 * right.a01 + left.a01 * right.a11;
        result.a10 = left.a10 * right.a00 + left.a11 * right.a10;
        result.a11 = left.a10 * right.a01 + left.a11 * right.a11;
        return result;
    }
};

// Test 1: shift_group_right by 1 on Matrix2x2
int
test_shift_group_right(sycl::queue& q, std::uint32_t sg_size)
{
    using Mat = Matrix2x2<std::int32_t>;
    const std::uint32_t n = sg_size;

    std::vector<Mat> input(n);
    for (std::uint32_t i = 0; i < n; ++i)
        input[i] = Mat(static_cast<std::int32_t>(i + 1), static_cast<std::int32_t>(i + 2));

    std::vector<Mat> output(n);

    {
        sycl::buffer<Mat> in_buf(input.data(), sycl::range<1>(n));
        sycl::buffer<Mat> out_buf(output.data(), sycl::range<1>(n));

        q.submit([&](sycl::handler& cgh) {
            auto in_acc = in_buf.template get_access<sycl::access::mode::read>(cgh);
            auto out_acc = out_buf.template get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(sycl::nd_range<1>(n, n), [=](sycl::nd_item<1> ndi) {
                auto sg = ndi.get_sub_group();
                std::uint32_t lid = sg.get_local_linear_id();
                Mat val = in_acc[lid];
                Mat shifted = sycl::shift_group_right(sg, val, 1);
                out_acc[lid] = shifted;
            });
        });
    }

    int errors = 0;
    for (std::uint32_t i = 1; i < n; ++i)
    {
        if (!(output[i] == input[i - 1]))
        {
            if (errors < 5)
                std::printf("  [shift] FAIL lane %u: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n", i, output[i].a00,
                            output[i].a01, output[i].a10, output[i].a11, input[i - 1].a00, input[i - 1].a01,
                            input[i - 1].a10, input[i - 1].a11);
            ++errors;
        }
    }
    return errors;
}

// Test 2: group_broadcast on Matrix2x2
int
test_group_broadcast(sycl::queue& q, std::uint32_t sg_size)
{
    using Mat = Matrix2x2<std::int32_t>;
    const std::uint32_t n = sg_size;

    Mat broadcast_val(42, 99);
    std::vector<Mat> output(n);

    {
        sycl::buffer<Mat> out_buf(output.data(), sycl::range<1>(n));

        q.submit([&](sycl::handler& cgh) {
            auto out_acc = out_buf.template get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(sycl::nd_range<1>(n, n), [=](sycl::nd_item<1> ndi) {
                auto sg = ndi.get_sub_group();
                std::uint32_t lid = sg.get_local_linear_id();
                Mat val(static_cast<std::int32_t>(lid), static_cast<std::int32_t>(lid + 10));
                if (lid == 0)
                    val = broadcast_val;
                Mat result = sycl::group_broadcast(sg, val, 0u);
                out_acc[lid] = result;
            });
        });
    }

    int errors = 0;
    for (std::uint32_t i = 0; i < n; ++i)
    {
        if (!(output[i] == broadcast_val))
        {
            if (errors < 5)
                std::printf("  [broadcast] FAIL lane %u: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n", i,
                            output[i].a00, output[i].a01, output[i].a10, output[i].a11, broadcast_val.a00,
                            broadcast_val.a01, broadcast_val.a10, broadcast_val.a11);
            ++errors;
        }
    }
    return errors;
}

// Test 3: Hillis-Steele inclusive scan using shift_group_right (single sub-group)
int
test_inclusive_scan_single_sg(sycl::queue& q, std::uint32_t sg_size)
{
    using Mat = Matrix2x2<std::int32_t>;
    multiply_matrix<std::int32_t> op;
    const std::uint32_t n = sg_size;

    std::vector<Mat> input(n);
    for (std::uint32_t i = 0; i < n; ++i)
        input[i] = Mat(static_cast<std::int32_t>(i), static_cast<std::int32_t>(i + 1));

    std::vector<Mat> expected(n);
    expected[0] = input[0];
    for (std::uint32_t i = 1; i < n; ++i)
        expected[i] = op(expected[i - 1], input[i]);

    std::vector<Mat> output(n);

    {
        sycl::buffer<Mat> in_buf(input.data(), sycl::range<1>(n));
        sycl::buffer<Mat> out_buf(output.data(), sycl::range<1>(n));

        q.submit([&](sycl::handler& cgh) {
            auto in_acc = in_buf.template get_access<sycl::access::mode::read>(cgh);
            auto out_acc = out_buf.template get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(sycl::nd_range<1>(n, n), [=](sycl::nd_item<1> ndi) {
                auto sg = ndi.get_sub_group();
                std::uint32_t lid = sg.get_local_linear_id();
                std::uint32_t sg_sz = sg.get_max_local_range()[0];

                Mat val = in_acc[lid];
                multiply_matrix<std::int32_t> mul;
                for (std::uint32_t shift = 1; shift <= sg_sz / 2; shift <<= 1)
                {
                    Mat partial = sycl::shift_group_right(sg, val, shift);
                    if (lid >= shift)
                        val = mul(partial, val);
                }
                out_acc[lid] = val;
            });
        });
    }

    int errors = 0;
    for (std::uint32_t i = 0; i < n; ++i)
    {
        if (!(output[i] == expected[i]))
        {
            if (errors < 5)
                std::printf("  [scan_1sg] FAIL lane %u: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n", i, output[i].a00,
                            output[i].a01, output[i].a10, output[i].a11, expected[i].a00, expected[i].a01,
                            expected[i].a10, expected[i].a11);
            ++errors;
        }
    }
    return errors;
}

// Test 4: Multi-subgroup inclusive scan (simulates reduce-then-scan phases)
int
test_multi_subgroup_scan(sycl::queue& q, std::uint32_t sg_size, std::size_t n)
{
    using Mat = Matrix2x2<std::int32_t>;
    multiply_matrix<std::int32_t> op;

    std::vector<Mat> input(n);
    for (std::size_t i = 0; i < n; ++i)
        input[i] = Mat(static_cast<std::int32_t>(i % 7), static_cast<std::int32_t>((i % 5) + 1));

    std::vector<Mat> expected(n);
    expected[0] = input[0];
    for (std::size_t i = 1; i < n; ++i)
        expected[i] = op(expected[i - 1], input[i]);

    std::vector<Mat> output(n, Mat(-1, -1));

    std::uint32_t wg_size = sg_size * 2;
    std::uint32_t num_wg = static_cast<std::uint32_t>((n + wg_size - 1) / wg_size);
    if (num_wg < 2)
        num_wg = 2;
    std::uint32_t total_items = num_wg * wg_size;
    std::uint32_t num_subgroups = num_wg * (wg_size / sg_size);

    std::vector<Mat> carries(num_subgroups);

    {
        sycl::buffer<Mat> in_buf(input.data(), sycl::range<1>(n));
        sycl::buffer<Mat> out_buf(output.data(), sycl::range<1>(n));
        sycl::buffer<Mat> carry_buf(carries.data(), sycl::range<1>(num_subgroups));

        q.submit([&](sycl::handler& cgh) {
            auto in_acc = in_buf.template get_access<sycl::access::mode::read>(cgh);
            auto out_acc = out_buf.template get_access<sycl::access::mode::write>(cgh);
            auto carry_acc = carry_buf.template get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(sycl::nd_range<1>(total_items, wg_size), [=](sycl::nd_item<1> ndi) {
                auto sg = ndi.get_sub_group();
                std::uint32_t global_id = ndi.get_global_linear_id();
                std::uint32_t lid = sg.get_local_linear_id();
                std::uint32_t sg_sz = sg.get_max_local_range()[0];
                std::uint32_t sg_global_id =
                    ndi.get_group(0) * (wg_size / sg_sz) + sg.get_group_linear_id();

                Mat val;
                bool in_range = global_id < n;
                if (in_range)
                    val = in_acc[global_id];
                else
                    val = Mat(); // identity

                multiply_matrix<std::int32_t> mul;
                for (std::uint32_t shift = 1; shift <= sg_sz / 2; shift <<= 1)
                {
                    Mat partial = sycl::shift_group_right(sg, val, shift);
                    if (lid >= shift)
                        val = mul(partial, val);
                }

                if (in_range)
                    out_acc[global_id] = val;

                Mat carry = sycl::group_broadcast(sg, val, sg_sz - 1);
                if (lid == 0)
                    carry_acc[sg_global_id] = carry;
            });
        });
    }

    // Scan carries on host and apply
    std::vector<Mat> carry_prefix(num_subgroups);
    carry_prefix[0] = carries[0];
    for (std::uint32_t i = 1; i < num_subgroups; ++i)
        carry_prefix[i] = op(carry_prefix[i - 1], carries[i]);

    for (std::size_t i = 0; i < n; ++i)
    {
        std::uint32_t sg_id = static_cast<std::uint32_t>(i / sg_size);
        if (sg_id > 0)
            output[i] = op(carry_prefix[sg_id - 1], output[i]);
    }

    int errors = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!(output[i] == expected[i]))
        {
            if (errors < 5)
                std::printf("  [multi_sg n=%zu] FAIL idx=%zu: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n", n, i,
                            output[i].a00, output[i].a01, output[i].a10, output[i].a11, expected[i].a00,
                            expected[i].a01, expected[i].a10, expected[i].a11);
            ++errors;
        }
    }
    return errors;
}

int
run_test()
{
    sycl::queue q;
    auto dev = q.get_device();
    std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("Driver: %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());

    auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    std::uint32_t sg_size = *std::max_element(sg_sizes.begin(), sg_sizes.end());
    std::printf("Sub-group size: %u, sizeof(Matrix2x2<int32_t>)=%zu\n\n", sg_size,
                sizeof(Matrix2x2<std::int32_t>));

    static_assert(std::is_trivially_copyable_v<Matrix2x2<std::int32_t>>);

    int total_errors = 0;

    std::printf("Test 1: shift_group_right\n");
    int err = test_shift_group_right(q, sg_size);
    std::printf("  %s (%d errors)\n", err ? "FAIL" : "PASS", err);
    total_errors += err;

    std::printf("Test 2: group_broadcast\n");
    err = test_group_broadcast(q, sg_size);
    std::printf("  %s (%d errors)\n", err ? "FAIL" : "PASS", err);
    total_errors += err;

    std::printf("Test 3: inclusive scan (1 sub-group)\n");
    err = test_inclusive_scan_single_sg(q, sg_size);
    std::printf("  %s (%d errors)\n", err ? "FAIL" : "PASS", err);
    total_errors += err;

    std::printf("Test 4: multi-subgroup scan (various sizes)\n");
    std::size_t test_sizes[] = {64, 100, 256, 500, 1000, 2048, 5000, 10000, 50000, 100000};
    for (std::size_t sz : test_sizes)
    {
        err = test_multi_subgroup_scan(q, sg_size, sz);
        if (err)
            std::printf("  n=%zu: FAIL (%d errors)\n", sz, err);
        total_errors += err;
    }
    if (total_errors == 0)
        std::printf("  PASS (all sizes)\n");

    // Repeat single-sg scan to catch intermittent failures
    std::printf("Test 5: inclusive scan repeated 50x (intermittency check)\n");
    for (int iter = 0; iter < 50; ++iter)
    {
        err = test_inclusive_scan_single_sg(q, sg_size);
        if (err)
        {
            std::printf("  FAIL on iteration %d (%d errors)\n", iter, err);
            total_errors += err;
            break;
        }
    }
    if (err == 0)
        std::printf("  PASS (50/50)\n");

    std::printf("\nTotal errors: %d\n", total_errors);
    return total_errors ? 1 : 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    int result = run_test();
    if (result != 0)
        return result;
#endif
    return TestUtils::done();
}
