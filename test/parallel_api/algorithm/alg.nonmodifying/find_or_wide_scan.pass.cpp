// -*- C++ -*-
//===-- find_or_wide_scan.pass.cpp ----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

// The find_or backend scans several contiguous elements per work item and votes once per batch of
// iterations, but only above a size threshold that every other test stays below, so nothing otherwise
// exercises that path. Force the threshold to zero and vary the match position: each position makes a
// different element of an iteration, and a different length of the growing batch, hold the match. A second
// match on the far side of the first distinguishes the forward tag from the backward one, which is what the
// shared early-exit vote has to get right.
#define _ONEDPL_FIND_OR_WIDE_SCAN_MIN_SIZE 0

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include <vector>

template <typename Policy>
void
test_at_size(Policy&& __exec, std::size_t __n)
{
    sycl::queue __q = __exec.queue();
    std::vector<int> __host(__n, 0);
    int* __d = sycl::malloc_device<int>(__n, __q);
    auto __is_one = [](int __x) { return __x == 1; };

    // Every position for a size a test can enumerate; a stride coprime with the scan width and the
    // work-group size above that, so slots and batch lengths are still hit at every alignment.
    const std::size_t __step = __n <= 1024 ? 1 : 37;

    for (std::size_t __pos = 0; __pos <= __n; __pos += __step)
    {
        std::fill(__host.begin(), __host.end(), 0);
        const bool __has_match = __pos < __n;
        if (__has_match)
            __host[__pos] = 1;
        // A decoy past the first match: a forward scan must ignore it, a backward scan must return it.
        const std::size_t __decoy = __has_match && __pos + 1 < __n ? __n - 1 : __pos;
        if (__has_match)
            __host[__decoy] = 1;
        __q.memcpy(__d, __host.data(), __n * sizeof(int)).wait();

        // Forward tag: the first match.
        EXPECT_TRUE(oneapi::dpl::find_if(__exec, __d, __d + __n, __is_one) == __d + __pos, "wrong index from find_if");
        // Backward tag: find_end over a one-element needle returns the last match.
        if (__n > 1)
        {
            const int __needle = 1;
            int* __nd = sycl::malloc_device<int>(1, __q);
            __q.memcpy(__nd, &__needle, sizeof(int)).wait();
            auto __expected = __has_match ? __d + __decoy : __d + __n;
            EXPECT_TRUE(oneapi::dpl::find_end(__exec, __d, __d + __n, __nd, __nd + 1) == __expected,
                        "wrong index from find_end");
            sycl::free(__nd, __q);
        }
        // Or tag: presence only.
        EXPECT_TRUE(oneapi::dpl::any_of(__exec, __d, __d + __n, __is_one) == __has_match, "wrong result from any_of");
        EXPECT_TRUE(oneapi::dpl::none_of(__exec, __d, __d + __n, __is_one) == !__has_match,
                    "wrong result from none_of");
        // Two ranges, so the scan loads from two streams per element.
        EXPECT_TRUE(oneapi::dpl::mismatch(__exec, __d, __d + __n, __d).first == __d + __n,
                    "wrong result from mismatch of a range with itself");
    }
    sycl::free(__d, __q);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto __policy = TestUtils::get_dpcpp_test_policy();
    // Sizes that are and are not a multiple of the scan width, spanning the batch-growth thresholds.
    for (std::size_t __n : {std::size_t(1), std::size_t(3), std::size_t(4), std::size_t(31), std::size_t(1024),
                            std::size_t(4095), std::size_t(1) << 14})
        test_at_size(__policy, __n);
#endif
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
