// -*- C++ -*-
//===-- rts_matrix_shuffle_vs_slm.pass.cpp ----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Compares native sycl::shift_group_right inclusive scan against SLM-based
// fallback for Matrix2x2<int32_t>. Any divergence between the two paths
// indicates a driver/compiler bug in multi-register sub-group shuffles.
// Uses the same size progression as the oneDPL matrix scan tests.
//
// Each kernel reports its own observed sub-group size since different kernels
// may get different sub-group sizes from the runtime.

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

using Mat = Matrix2x2<std::int32_t>;

// Inclusive scan using native shift_group_right.
// Writes sub-group-local inclusive scan results. Reports observed sg_size.
void
kernel_native_scan(sycl::queue& q, const std::vector<Mat>& input, std::vector<Mat>& output, std::uint32_t wg_size,
                   std::uint32_t& observed_sg_size)
{
    std::size_t n = input.size();
    std::uint32_t total_items = static_cast<std::uint32_t>(((n + wg_size - 1) / wg_size) * wg_size);

    sycl::buffer<Mat> in_buf(input.data(), sycl::range<1>(n));
    sycl::buffer<Mat> out_buf(output.data(), sycl::range<1>(n));
    sycl::buffer<std::uint32_t> sg_buf(&observed_sg_size, sycl::range<1>(1));

    q.submit([&](sycl::handler& cgh) {
        auto in_acc = in_buf.template get_access<sycl::access::mode::read>(cgh);
        auto out_acc = out_buf.template get_access<sycl::access::mode::write>(cgh);
        auto sg_acc = sg_buf.template get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::nd_range<1>(total_items, wg_size), [=](sycl::nd_item<1> ndi) {
            auto sg = ndi.get_sub_group();
            std::uint32_t global_id = ndi.get_global_linear_id();
            std::uint32_t lid = sg.get_local_linear_id();
            std::uint32_t sg_sz = sg.get_max_local_range()[0];

            Mat val;
            bool in_range = global_id < n;
            if (in_range)
                val = in_acc[global_id];
            else
                val = Mat();

            multiply_matrix<std::int32_t> mul;
            for (std::uint32_t shift = 1; shift <= sg_sz / 2; shift <<= 1)
            {
                Mat partial = sycl::shift_group_right(sg, val, shift);
                if (lid >= shift)
                    val = mul(partial, val);
            }

            if (in_range)
                out_acc[global_id] = val;
            if (global_id == 0)
                sg_acc[0] = sg_sz;
        });
    });
    q.wait();
}

// Inclusive scan using SLM-based shift (the fallback path oneDPL uses for non-trivially-copyable).
// Reports observed sg_size.
void
kernel_slm_scan(sycl::queue& q, const std::vector<Mat>& input, std::vector<Mat>& output, std::uint32_t wg_size,
                std::uint32_t& observed_sg_size)
{
    std::size_t n = input.size();
    std::uint32_t total_items = static_cast<std::uint32_t>(((n + wg_size - 1) / wg_size) * wg_size);

    sycl::buffer<Mat> in_buf(input.data(), sycl::range<1>(n));
    sycl::buffer<Mat> out_buf(output.data(), sycl::range<1>(n));
    sycl::buffer<std::uint32_t> sg_buf(&observed_sg_size, sycl::range<1>(1));

    q.submit([&](sycl::handler& cgh) {
        auto in_acc = in_buf.template get_access<sycl::access::mode::read>(cgh);
        auto out_acc = out_buf.template get_access<sycl::access::mode::write>(cgh);
        auto sg_acc = sg_buf.template get_access<sycl::access::mode::write>(cgh);
        sycl::local_accessor<Mat> slm(wg_size, cgh);
        cgh.parallel_for(sycl::nd_range<1>(total_items, wg_size), [=](sycl::nd_item<1> ndi) {
            auto sg = ndi.get_sub_group();
            std::uint32_t global_id = ndi.get_global_linear_id();
            std::uint32_t lid = sg.get_local_linear_id();
            std::uint32_t sg_sz = sg.get_max_local_range()[0];
            std::uint32_t sg_base = sg.get_group_linear_id() * sg_sz;

            Mat val;
            bool in_range = global_id < n;
            if (in_range)
                val = in_acc[global_id];
            else
                val = Mat();

            multiply_matrix<std::int32_t> mul;
            for (std::uint32_t shift = 1; shift <= sg_sz / 2; shift <<= 1)
            {
                // SLM-based shift_group_right
                slm[sg_base + lid] = val;
                sycl::group_barrier(sg);
                Mat partial = slm[sg_base + ((lid >= shift) ? lid - shift : lid)];
                sycl::group_barrier(sg);

                if (lid >= shift)
                    val = mul(partial, val);
            }

            if (in_range)
                out_acc[global_id] = val;
            if (global_id == 0)
                sg_acc[0] = sg_sz;
        });
    });
    q.wait();
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

    std::uint32_t max_sg_size = *std::max_element(sg_sizes.begin(), sg_sizes.end());
    std::uint32_t wg_size = max_sg_size * 4;
    std::printf("Work-group size: %u\n", wg_size);
    std::printf("sizeof(Matrix2x2<int32_t>) = %zu\n\n", sizeof(Mat));

    static_assert(std::is_trivially_copyable_v<Mat>);

    int total_failures = 0;
    bool sg_size_printed = false;

    // Use the same size progression as the oneDPL inclusive_scan_matrix.pass test
    std::vector<std::size_t> test_sizes;
    for (std::size_t sz = 1; sz <= 16; ++sz)
        test_sizes.push_back(sz);
    for (double sz = 16.0; sz <= 100000; sz *= 3.1415)
        test_sizes.push_back(static_cast<std::size_t>(sz));

    for (std::size_t n : test_sizes)
    {
        std::vector<Mat> input(n);
        for (std::size_t i = 0; i < n; ++i)
            input[i] = Mat(static_cast<std::int32_t>(i % 11), static_cast<std::int32_t>((i % 7) + 1));

        std::vector<Mat> out_native(n);
        std::vector<Mat> out_slm(n);
        std::uint32_t native_sg_size = 0;
        std::uint32_t slm_sg_size = 0;

        kernel_native_scan(q, input, out_native, wg_size, native_sg_size);
        kernel_slm_scan(q, input, out_slm, wg_size, slm_sg_size);

        if (!sg_size_printed)
        {
            std::printf("Native kernel sg_size: %u, SLM kernel sg_size: %u\n\n", native_sg_size, slm_sg_size);
            sg_size_printed = true;
        }

        if (native_sg_size != slm_sg_size)
        {
            std::printf("  WARNING n=%zu: kernels got different sg_sizes (native=%u, slm=%u) — cannot compare\n", n,
                        native_sg_size, slm_sg_size);
            continue;
        }

        std::uint32_t mismatch_count = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (!(out_native[i] == out_slm[i]))
            {
                if (mismatch_count < 3)
                {
                    std::printf("  MISMATCH n=%zu idx=%zu (sg=%zu lane=%zu): native={%d,%d,%d,%d} slm={%d,%d,%d,%d}\n",
                                n, i, i / native_sg_size, i % native_sg_size, out_native[i].a00, out_native[i].a01,
                                out_native[i].a10, out_native[i].a11, out_slm[i].a00, out_slm[i].a01, out_slm[i].a10,
                                out_slm[i].a11);
                }
                ++mismatch_count;
            }
        }

        if (mismatch_count > 0)
        {
            std::printf("  FAIL n=%zu: %u mismatches between native and SLM paths\n", n, mismatch_count);
            ++total_failures;
        }
    }

    std::printf("\n%zu sizes tested, %d failures\n", test_sizes.size(), total_failures);
    if (total_failures > 0)
    {
        std::printf("DIVERGENCE: native sub-group shuffle produces different results than\n"
                    "SLM-based communication for Matrix2x2<int32_t>.\n");
    }

    return total_failures ? 1 : 0;
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
