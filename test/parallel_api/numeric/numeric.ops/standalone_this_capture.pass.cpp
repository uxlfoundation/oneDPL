// -*- C++ -*-
//===-- standalone_this_capture.pass.cpp ----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Standalone SYCL test: Tests the [=, *this] capture of a large struct
// containing multiple members matching the RTS scan submitter layout.
// The kernel body is trivial — just reads from the captured struct and
// writes to output. If this crashes, the *this capture size is the issue.

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <functional>

// Match the approximate layout of __parallel_reduce_then_scan_scan_submitter
template <typename _ReduceOp, typename _InitType>
struct big_submitter
{
    sycl::event
    operator()(sycl::queue& __q, _InitType* __out_usm, std::size_t __n) const
    {
        return __q.submit([&, *this](sycl::handler& __cgh) {
            sycl::local_accessor<_InitType, 1> __sub_group_partials(__max_num_sub_groups_local, __cgh);
            sycl::local_accessor<_InitType, 1> __comm_slm(__use_slm_for_comm ? __work_group_size : 0, __cgh);

            __cgh.parallel_for<class big_capture_kernel>(
                sycl::nd_range<1>(sycl::range<1>(__work_group_size * __max_num_work_groups),
                                  sycl::range<1>(__work_group_size)),
                [=, *this](sycl::nd_item<1> __ndi) {
                    std::uint32_t __gid = __ndi.get_global_linear_id();
                    std::uint32_t __lid = __ndi.get_local_linear_id();

                    // Read from all captured members to ensure they're all live
                    _InitType __val = __init;
                    __val = __reduce_op(__val, static_cast<_InitType>(__max_num_work_groups));
                    __val = __reduce_op(__val, static_cast<_InitType>(__work_group_size));
                    __val = __reduce_op(__val, static_cast<_InitType>(__max_block_size));
                    __val = __reduce_op(__val, static_cast<_InitType>(__max_num_sub_groups_local));
                    __val = __reduce_op(__val, static_cast<_InitType>(__max_num_sub_groups_global));
                    __val = __reduce_op(__val, static_cast<_InitType>(__num_blocks));
                    __val = __reduce_op(__val, static_cast<_InitType>(__n_stored));
                    __val = __reduce_op(__val, static_cast<_InitType>(__use_slm_for_comm ? 1 : 0));
                    __val = __reduce_op(__val, static_cast<_InitType>(__dummy_field_1));
                    __val = __reduce_op(__val, static_cast<_InitType>(__dummy_field_2));
                    __val = __reduce_op(__val, static_cast<_InitType>(__dummy_field_3));

                    // Use SLM
                    if (__use_slm_for_comm && __lid < __work_group_size)
                        __comm_slm[__lid] = __val;
                    sycl::group_barrier(__ndi.get_group());

                    auto __sg = __ndi.get_sub_group();
                    if (__sg.get_local_linear_id() == 0)
                        __sub_group_partials[__sg.get_group_linear_id()] = __val;
                    sycl::group_barrier(__ndi.get_group());

                    if (__gid < __n)
                        __out_usm[__gid] = __val;
                });
        });
    }

    // Match scan submitter member layout
    const std::uint32_t __max_num_work_groups;
    const std::uint32_t __work_group_size;
    const std::uint32_t __max_block_size;
    const std::uint32_t __max_num_sub_groups_local;
    const std::uint32_t __max_num_sub_groups_global;
    const std::size_t __num_blocks;
    const std::size_t __n_stored;
    const bool __use_slm_for_comm;

    const _ReduceOp __reduce_op;
    const _InitType __init;

    // Extra padding to increase captured struct size
    const std::uint64_t __dummy_field_1;
    const std::uint64_t __dummy_field_2;
    const std::uint64_t __dummy_field_3;
};

int
run_this_capture()
{
    sycl::queue q;
    constexpr std::size_t n = 2048;

    auto* out_usm = sycl::malloc_device<std::int32_t>(n, q);

    big_submitter<std::plus<std::int32_t>, std::int32_t> sub{
        8, 256, 256 * 128 * 8, 8, 64, 1, n, true,
        std::plus<std::int32_t>{}, 0, 42, 43, 44};

    auto ev = sub(q, out_usm, n);
    ev.wait();

    std::vector<std::int32_t> output(n, 0);
    q.memcpy(output.data(), out_usm, n * sizeof(std::int32_t)).wait();
    sycl::free(out_usm, q);

    bool ok = false;
    for (std::size_t i = 0; i < n; ++i)
        if (output[i] != 0) { ok = true; break; }

    if (!ok)
    {
        std::printf("FAIL: standalone_this_capture output is all zeros\n");
        return 1;
    }
    return 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    int result = run_this_capture();
    if (result != 0)
        return result;
#endif
    return TestUtils::done();
}
