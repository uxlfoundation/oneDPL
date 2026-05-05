// -*- C++ -*-
//===-- rts_matrix_cpu_repro.pass.cpp ---------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Minimal standalone reproducer: inclusive scan of a 16-byte struct (2x2 matrix)
// using SLM-based sub-group communication in a reduce-then-scan pattern.
// Crashes (SegFault / 0xc0000374) on Windows CPU in release mode with certain
// Intel OpenCL CPU runtime drivers.
//
// Build (Linux):   icpx -fsycl -O2 -o rts_matrix_cpu_repro rts_matrix_cpu_repro.pass.cpp
// Build (Windows): icx-cl /EHsc -fsycl -O2 rts_matrix_cpu_repro.pass.cpp
//
// No external dependencies beyond SYCL.

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <vector>

struct Mat
{
    std::int32_t a00, a01, a10, a11;
};

Mat identity()
{
    return {1, 0, 0, 1};
}

Mat make_mat(std::int32_t k)
{
    return {0, k, k, k + 1};
}

Mat multiply(Mat l, Mat r)
{
    return {l.a00 * r.a00 + l.a01 * r.a10, l.a00 * r.a01 + l.a01 * r.a11,
            l.a10 * r.a00 + l.a11 * r.a10, l.a10 * r.a01 + l.a11 * r.a11};
}

bool equal(Mat a, Mat b)
{
    return a.a00 == b.a00 && a.a01 == b.a01 && a.a10 == b.a10 && a.a11 == b.a11;
}

// SLM-based shift right within a sub-group
Mat slm_shift_right(sycl::sub_group sg, Mat val, std::uint32_t shift, Mat* slm)
{
    std::uint32_t lid = sg.get_local_linear_id();
    std::uint32_t base = sg.get_group_linear_id() * sg.get_max_local_range()[0];
    slm[base + lid] = val;
    sycl::group_barrier(sg);
    Mat result = slm[base + ((lid >= shift) ? lid - shift : lid)];
    sycl::group_barrier(sg);
    return result;
}

// SLM-based broadcast from a specific lane
Mat slm_broadcast(sycl::sub_group sg, Mat val, std::uint32_t src, Mat* slm)
{
    std::uint32_t lid = sg.get_local_linear_id();
    std::uint32_t base = sg.get_group_linear_id() * sg.get_max_local_range()[0];
    slm[base + lid] = val;
    sycl::group_barrier(sg);
    Mat result = slm[base + src];
    sycl::group_barrier(sg);
    return result;
}

int
run_test()
{
    sycl::queue q;
    auto dev = q.get_device();
    std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("Driver: %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("is_cpu: %d\n", dev.is_cpu());

    constexpr std::size_t N = 20000;
    const std::uint32_t wg_size = dev.is_cpu() ? 64 : 128;

    // Prepare input
    std::vector<Mat> input(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = make_mat(static_cast<std::int32_t>((i % 7) + 1));

    // Host reference inclusive scan
    std::vector<Mat> expected(N);
    expected[0] = input[0];
    for (std::size_t i = 1; i < N; ++i)
        expected[i] = multiply(expected[i - 1], input[i]);

    // Device output
    std::vector<Mat> output(N);
    for (auto& m : output)
        m = {-1, -1, -1, -1};

    std::uint32_t total_items = static_cast<std::uint32_t>(((N + wg_size - 1) / wg_size) * wg_size);

    // Allocate space for per-sub-group carry-outs (over-allocate for safety)
    std::vector<Mat> carries(total_items);
    std::uint32_t observed_sg_size = 0;

    // Phase 1: Sub-group local inclusive scan + carry extraction
    {
        sycl::buffer<Mat> in_buf(input.data(), sycl::range<1>(N));
        sycl::buffer<Mat> out_buf(output.data(), sycl::range<1>(N));
        sycl::buffer<Mat> carry_buf(carries.data(), sycl::range<1>(total_items));
        sycl::buffer<std::uint32_t> sg_buf(&observed_sg_size, sycl::range<1>(1));

        q.submit([&](sycl::handler& cgh) {
            auto in = in_buf.get_access<sycl::access::mode::read>(cgh);
            auto out = out_buf.get_access<sycl::access::mode::write>(cgh);
            auto car = carry_buf.get_access<sycl::access::mode::write>(cgh);
            auto sg_acc = sg_buf.get_access<sycl::access::mode::write>(cgh);
            sycl::local_accessor<Mat> slm(wg_size, cgh);

            cgh.parallel_for(sycl::nd_range<1>(total_items, wg_size), [=](sycl::nd_item<1> ndi) {
                auto sg = ndi.get_sub_group();
                std::uint32_t gid = ndi.get_global_linear_id();
                std::uint32_t lid = sg.get_local_linear_id();
                std::uint32_t sg_sz = sg.get_max_local_range()[0];
                std::uint32_t sg_global = ndi.get_group(0) * (wg_size / sg_sz) + sg.get_group_linear_id();
                Mat* slm_ptr = &slm[0];

                // Load input (identity for out-of-range)
                Mat val = (gid < N) ? in[gid] : Mat{1, 0, 0, 1};

                // Hillis-Steele inclusive scan using SLM communication
                for (std::uint32_t shift = 1; shift <= sg_sz / 2; shift <<= 1)
                {
                    Mat shifted = slm_shift_right(sg, val, shift, slm_ptr);
                    if (lid >= shift)
                    {
                        val = {shifted.a00 * val.a00 + shifted.a01 * val.a10,
                               shifted.a00 * val.a01 + shifted.a01 * val.a11,
                               shifted.a10 * val.a00 + shifted.a11 * val.a10,
                               shifted.a10 * val.a01 + shifted.a11 * val.a11};
                    }
                }

                // Write scan result
                if (gid < N)
                    out[gid] = val;

                // Extract carry (broadcast last lane's value)
                Mat carry = slm_broadcast(sg, val, sg_sz - 1, slm_ptr);
                if (lid == 0)
                    car[sg_global] = carry;

                if (gid == 0)
                    sg_acc[0] = sg_sz;
            });
        });
    }

    std::printf("Runtime sub-group size: %u, work-group size: %u\n", observed_sg_size, wg_size);

    // Phase 2: Host-side carry propagation
    std::uint32_t num_sgs = total_items / observed_sg_size;
    std::vector<Mat> carry_prefix(num_sgs);
    carry_prefix[0] = carries[0];
    for (std::uint32_t i = 1; i < num_sgs; ++i)
        carry_prefix[i] = multiply(carry_prefix[i - 1], carries[i]);

    for (std::size_t i = 0; i < N; ++i)
    {
        std::uint32_t sg_id = static_cast<std::uint32_t>(i / observed_sg_size);
        if (sg_id > 0)
            output[i] = multiply(carry_prefix[sg_id - 1], output[i]);
    }

    // Verify
    int errors = 0;
    for (std::size_t i = 0; i < N; ++i)
    {
        if (!equal(output[i], expected[i]))
        {
            if (errors < 10)
                std::printf("  FAIL [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n", i, output[i].a00,
                            output[i].a01, output[i].a10, output[i].a11, expected[i].a00, expected[i].a01,
                            expected[i].a10, expected[i].a11);
            ++errors;
        }
    }

    std::printf("%s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);
    return errors ? 1 : 0;
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
