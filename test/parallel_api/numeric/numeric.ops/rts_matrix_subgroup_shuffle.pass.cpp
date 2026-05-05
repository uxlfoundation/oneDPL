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
//
// Each kernel reports its own observed sub-group size back to the host for
// correct validation (different kernels may get different sub-group sizes).

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
// Kernel reports its own sg_size. Validation checks within each sub-group boundary.
int
test_shift_group_right(sycl::queue& q, std::uint32_t wg_size)
{
    using Mat = Matrix2x2<std::int32_t>;

    std::vector<Mat> input(wg_size);
    for (std::uint32_t i = 0; i < wg_size; ++i)
        input[i] = Mat(static_cast<std::int32_t>(i + 1), static_cast<std::int32_t>(i + 2));

    std::vector<Mat> output(wg_size);
    std::uint32_t observed_sg_size = 0;

    {
        sycl::buffer<Mat> in_buf(input.data(), sycl::range<1>(wg_size));
        sycl::buffer<Mat> out_buf(output.data(), sycl::range<1>(wg_size));
        sycl::buffer<std::uint32_t> sg_buf(&observed_sg_size, sycl::range<1>(1));

        q.submit([&](sycl::handler& cgh) {
            auto in_acc = in_buf.template get_access<sycl::access::mode::read>(cgh);
            auto out_acc = out_buf.template get_access<sycl::access::mode::write>(cgh);
            auto sg_acc = sg_buf.template get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(sycl::nd_range<1>(wg_size, wg_size), [=](sycl::nd_item<1> ndi) {
                auto sg = ndi.get_sub_group();
                std::uint32_t global_id = ndi.get_global_linear_id();
                Mat val = in_acc[global_id];
                Mat shifted = sycl::shift_group_right(sg, val, 1);
                out_acc[global_id] = shifted;
                if (global_id == 0)
                    sg_acc[0] = sg.get_max_local_range()[0];
            });
        });
    }

    std::printf("  (kernel sg_size=%u)\n", observed_sg_size);

    int errors = 0;
    std::uint32_t num_sgs = wg_size / observed_sg_size;
    for (std::uint32_t sg = 0; sg < num_sgs; ++sg)
    {
        std::uint32_t base = sg * observed_sg_size;
        // Lane 0 of each sub-group: undefined after shift_right(1), skip it
        for (std::uint32_t lid = 1; lid < observed_sg_size; ++lid)
        {
            std::uint32_t i = base + lid;
            if (!(output[i] == input[i - 1]))
            {
                if (errors < 5)
                    std::printf("  [shift] FAIL sg=%u lane=%u (idx=%u): got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n", sg,
                                lid, i, output[i].a00, output[i].a01, output[i].a10, output[i].a11, input[i - 1].a00,
                                input[i - 1].a01, input[i - 1].a10, input[i - 1].a11);
                ++errors;
            }
        }
    }
    return errors;
}

// Test 2: group_broadcast on Matrix2x2
int
test_group_broadcast(sycl::queue& q, std::uint32_t wg_size)
{
    using Mat = Matrix2x2<std::int32_t>;

    std::vector<Mat> output(wg_size);
    std::uint32_t observed_sg_size = 0;

    {
        sycl::buffer<Mat> out_buf(output.data(), sycl::range<1>(wg_size));
        sycl::buffer<std::uint32_t> sg_buf(&observed_sg_size, sycl::range<1>(1));

        q.submit([&](sycl::handler& cgh) {
            auto out_acc = out_buf.template get_access<sycl::access::mode::write>(cgh);
            auto sg_acc = sg_buf.template get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(sycl::nd_range<1>(wg_size, wg_size), [=](sycl::nd_item<1> ndi) {
                auto sg = ndi.get_sub_group();
                std::uint32_t global_id = ndi.get_global_linear_id();
                // Each lane gets a unique value based on global_id
                Mat val(static_cast<std::int32_t>(global_id + 42), static_cast<std::int32_t>(global_id + 99));
                Mat result = sycl::group_broadcast(sg, val, 0u);
                out_acc[global_id] = result;
                if (global_id == 0)
                    sg_acc[0] = sg.get_max_local_range()[0];
            });
        });
    }

    std::printf("  (kernel sg_size=%u)\n", observed_sg_size);

    int errors = 0;
    std::uint32_t num_sgs = wg_size / observed_sg_size;
    for (std::uint32_t sg = 0; sg < num_sgs; ++sg)
    {
        std::uint32_t base = sg * observed_sg_size;
        // Lane 0 of this sub-group has global_id=base, so its value is:
        Mat expected_val(static_cast<std::int32_t>(base + 42), static_cast<std::int32_t>(base + 99));
        for (std::uint32_t lid = 0; lid < observed_sg_size; ++lid)
        {
            std::uint32_t i = base + lid;
            if (!(output[i] == expected_val))
            {
                if (errors < 5)
                    std::printf("  [broadcast] FAIL sg=%u lane=%u (idx=%u): got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                                sg, lid, i, output[i].a00, output[i].a01, output[i].a10, output[i].a11,
                                expected_val.a00, expected_val.a01, expected_val.a10, expected_val.a11);
                ++errors;
            }
        }
    }
    return errors;
}

// Test 3: Hillis-Steele inclusive scan with carry propagation (full scan pattern)
// Kernel does sub-group-local scan + carry extraction. Host applies carries.
// Kernel reports its actual sg_size for correct host-side validation.
int
test_full_scan(sycl::queue& q, std::uint32_t wg_size, std::size_t n)
{
    using Mat = Matrix2x2<std::int32_t>;
    multiply_matrix<std::int32_t> op;

    std::vector<Mat> input(n);
    for (std::size_t i = 0; i < n; ++i)
        input[i] = Mat(static_cast<std::int32_t>(i % 7), static_cast<std::int32_t>((i % 5) + 1));

    // Reference: global inclusive scan
    std::vector<Mat> expected(n);
    expected[0] = input[0];
    for (std::size_t i = 1; i < n; ++i)
        expected[i] = op(expected[i - 1], input[i]);

    std::vector<Mat> output(n, Mat(-1, -1));
    std::uint32_t total_items = static_cast<std::uint32_t>(((n + wg_size - 1) / wg_size) * wg_size);
    // Allocate carries for worst case (smallest possible sg_size = 1 per item)
    // We'll only use the first num_subgroups entries based on actual sg_size
    std::vector<Mat> carries(total_items);
    std::uint32_t observed_sg_size = 0;

    {
        sycl::buffer<Mat> in_buf(input.data(), sycl::range<1>(n));
        sycl::buffer<Mat> out_buf(output.data(), sycl::range<1>(n));
        sycl::buffer<Mat> carry_buf(carries.data(), sycl::range<1>(total_items));
        sycl::buffer<std::uint32_t> sg_buf(&observed_sg_size, sycl::range<1>(1));

        q.submit([&](sycl::handler& cgh) {
            auto in_acc = in_buf.template get_access<sycl::access::mode::read>(cgh);
            auto out_acc = out_buf.template get_access<sycl::access::mode::write>(cgh);
            auto carry_acc = carry_buf.template get_access<sycl::access::mode::write>(cgh);
            auto sg_acc = sg_buf.template get_access<sycl::access::mode::write>(cgh);
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

                // Extract carry (last lane's inclusive scan value)
                Mat carry = sycl::group_broadcast(sg, val, sg_sz - 1);
                if (lid == 0)
                    carry_acc[sg_global_id] = carry;

                if (global_id == 0)
                    sg_acc[0] = sg_sz;
            });
        });
    }

    // Host-side carry prefix sum and apply using the kernel's actual sg_size
    std::uint32_t num_subgroups = total_items / observed_sg_size;
    std::vector<Mat> carry_prefix(num_subgroups);
    carry_prefix[0] = carries[0];
    for (std::uint32_t i = 1; i < num_subgroups; ++i)
        carry_prefix[i] = op(carry_prefix[i - 1], carries[i]);

    for (std::size_t i = 0; i < n; ++i)
    {
        std::uint32_t sg_id = static_cast<std::uint32_t>(i / observed_sg_size);
        if (sg_id > 0)
            output[i] = op(carry_prefix[sg_id - 1], output[i]);
    }

    int errors = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!(output[i] == expected[i]))
        {
            if (errors < 5)
                std::printf("  [full_scan n=%zu sg_sz=%u] FAIL idx=%zu: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n", n,
                            observed_sg_size, i, output[i].a00, output[i].a01, output[i].a10, output[i].a11,
                            expected[i].a00, expected[i].a01, expected[i].a10, expected[i].a11);
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
    std::printf("Reported sub-group sizes:");
    for (auto s : sg_sizes)
        std::printf(" %zu", s);
    std::printf("\n");
    std::printf("sizeof(Matrix2x2<int32_t>) = %zu\n", sizeof(Matrix2x2<std::int32_t>));

    static_assert(std::is_trivially_copyable_v<Matrix2x2<std::int32_t>>);

    std::uint32_t max_sg_size = *std::max_element(sg_sizes.begin(), sg_sizes.end());
    std::uint32_t wg_size = max_sg_size * 4;
    std::printf("Work-group size: %u\n\n", wg_size);

    int total_errors = 0;

    std::printf("Test 1: shift_group_right\n");
    int err = test_shift_group_right(q, wg_size);
    std::printf("  %s (%d errors)\n", err ? "FAIL" : "PASS", err);
    total_errors += err;

    std::printf("Test 2: group_broadcast\n");
    err = test_group_broadcast(q, wg_size);
    std::printf("  %s (%d errors)\n", err ? "FAIL" : "PASS", err);
    total_errors += err;

    std::printf("Test 3: full scan with carry propagation (various sizes)\n");
    int test3_errors = 0;
    std::size_t full_sizes[] = {64, 100, 256, 500, 1000, 2048, 5000, 10000, 50000, 100000};
    for (std::size_t sz : full_sizes)
    {
        err = test_full_scan(q, wg_size, sz);
        if (err)
            std::printf("  n=%zu: FAIL (%d errors)\n", sz, err);
        test3_errors += err;
    }
    if (test3_errors == 0)
        std::printf("  PASS (all sizes)\n");
    total_errors += test3_errors;

    // Repeat to catch intermittent failures
    std::printf("Test 4: full scan repeated 50x with n=1000 (intermittency check)\n");
    int test4_errors = 0;
    for (int iter = 0; iter < 50; ++iter)
    {
        err = test_full_scan(q, wg_size, 1000);
        if (err)
        {
            std::printf("  FAIL on iteration %d (%d errors)\n", iter, err);
            test4_errors += err;
            break;
        }
    }
    if (test4_errors == 0)
        std::printf("  PASS (50/50)\n");
    total_errors += test4_errors;

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
