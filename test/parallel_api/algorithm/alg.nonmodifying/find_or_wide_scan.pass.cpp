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

// The find_or backend scans several contiguous elements per work item, but only above a size threshold that
// every other test stays below, so nothing otherwise exercises that path. Force the threshold to zero and
// vary the match position, which puts the match at every element of an iteration and at every alignment
// against the work-group size. A second match on the far side of the first distinguishes the forward tag
// from the backward one, which is what the shared early-exit vote has to get right.
//
// Not covered here: batches longer than one iteration. The batch length only grows past 1 above roughly 64M
// elements per launch, which no test can afford, so the multi-iteration vote is covered by benchmark runs
// and not by this test.
#define _ONEDPL_FIND_OR_WIDE_SCAN_MIN_SIZE 0

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include <cstdint>
#    include <vector>

// One name per call below: these algorithms share the find_or kernels, so under explicit kernel names a
// single policy would give every call the same kernel name.
class __find_if_name;
class __find_end_name;
class __any_of_name;
class __none_of_name;
class __mismatch_name;

template <typename Policy>
void
test_at_size(Policy&& __exec, std::size_t __n)
{
    sycl::queue __q = __exec.queue();
    auto __exec_find_if = TestUtils::make_new_policy<__find_if_name>(__exec);
    auto __exec_find_end = TestUtils::make_new_policy<__find_end_name>(__exec);
    auto __exec_any_of = TestUtils::make_new_policy<__any_of_name>(__exec);
    auto __exec_none_of = TestUtils::make_new_policy<__none_of_name>(__exec);
    auto __exec_mismatch = TestUtils::make_new_policy<__mismatch_name>(__exec);
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
        EXPECT_TRUE(oneapi::dpl::find_if(__exec_find_if, __d, __d + __n, __is_one) == __d + __pos,
                    "wrong index from find_if");
        // Backward tag: find_end over a one-element needle returns the last match.
        if (__n > 1)
        {
            const int __needle = 1;
            int* __nd = sycl::malloc_device<int>(1, __q);
            __q.memcpy(__nd, &__needle, sizeof(int)).wait();
            auto __expected = __has_match ? __d + __decoy : __d + __n;
            EXPECT_TRUE(oneapi::dpl::find_end(__exec_find_end, __d, __d + __n, __nd, __nd + 1) == __expected,
                        "wrong index from find_end");
            sycl::free(__nd, __q);
        }
        // Or tag: presence only.
        EXPECT_TRUE(oneapi::dpl::any_of(__exec_any_of, __d, __d + __n, __is_one) == __has_match,
                    "wrong result from any_of");
        EXPECT_TRUE(oneapi::dpl::none_of(__exec_none_of, __d, __d + __n, __is_one) == !__has_match,
                    "wrong result from none_of");
        // Two ranges, so the scan loads from two streams per element.
        EXPECT_TRUE(oneapi::dpl::mismatch(__exec_mismatch, __d, __d + __n, __d).first == __d + __n,
                    "wrong result from mismatch of a range with itself");
    }
    sycl::free(__d, __q);
}

// Two 8-byte ranges over the wide scan; no other test reaches this width above the size threshold.
class __mismatch_8byte_name;
class __equal_8byte_name;

template <typename Policy>
void
test_two_8byte_ranges(Policy&& __exec, std::size_t __n)
{
    using _T = std::uint64_t;
    sycl::queue __q = __exec.queue();
    auto __exec_mismatch = TestUtils::make_new_policy<__mismatch_8byte_name>(__exec);
    auto __exec_equal = TestUtils::make_new_policy<__equal_8byte_name>(__exec);
    std::vector<_T> __host(__n, _T(1));
    _T* __d1 = sycl::malloc_device<_T>(__n, __q);
    _T* __d2 = sycl::malloc_device<_T>(__n, __q);
    __q.memcpy(__d1, __host.data(), __n * sizeof(_T)).wait();

    const std::size_t __step = __n <= 1024 ? 1 : 37;

    for (std::size_t __pos = 0; __pos <= __n; __pos += __step)
    {
        std::fill(__host.begin(), __host.end(), _T(1));
        const bool __differs = __pos < __n;
        if (__differs)
            __host[__pos] = _T(2);
        __q.memcpy(__d2, __host.data(), __n * sizeof(_T)).wait();

        EXPECT_TRUE(oneapi::dpl::mismatch(__exec_mismatch, __d1, __d1 + __n, __d2).first ==
                        (__differs ? __d1 + __pos : __d1 + __n),
                    "wrong index from mismatch of two 8-byte ranges");
        EXPECT_TRUE(oneapi::dpl::equal(__exec_equal, __d1, __d1 + __n, __d2) == !__differs,
                    "wrong result from equal of two 8-byte ranges");
    }
    sycl::free(__d1, __q);
    sycl::free(__d2, __q);
}

// A 2-byte element type; the cases above cover only 4 and 8 bytes. Only any_of and none_of take the wide
// scan at this width -- find_if, mismatch and equal are gated to the narrow one, and cover it here.
class __find_if_2byte_name;
class __any_of_2byte_name;
class __none_of_2byte_name;
class __mismatch_2byte_name;
class __equal_2byte_name;

template <typename Policy>
void
test_2byte_ranges(Policy&& __exec, std::size_t __n)
{
    using _T = std::uint16_t;
    sycl::queue __q = __exec.queue();
    auto __exec_find_if = TestUtils::make_new_policy<__find_if_2byte_name>(__exec);
    auto __exec_any_of = TestUtils::make_new_policy<__any_of_2byte_name>(__exec);
    auto __exec_none_of = TestUtils::make_new_policy<__none_of_2byte_name>(__exec);
    auto __exec_mismatch = TestUtils::make_new_policy<__mismatch_2byte_name>(__exec);
    auto __exec_equal = TestUtils::make_new_policy<__equal_2byte_name>(__exec);
    auto __is_one = [](_T __x) { return __x == _T(1); };
    std::vector<_T> __host(__n, _T(0));
    _T* __d1 = sycl::malloc_device<_T>(__n, __q);
    _T* __d2 = sycl::malloc_device<_T>(__n, __q);
    // The second range stays all-zero, so one match in the first serves every tag below.
    __q.memcpy(__d2, __host.data(), __n * sizeof(_T)).wait();

    const std::size_t __step = __n <= 1024 ? 1 : 37;

    for (std::size_t __pos = 0; __pos <= __n; __pos += __step)
    {
        std::fill(__host.begin(), __host.end(), _T(0));
        const bool __has_match = __pos < __n;
        if (__has_match)
            __host[__pos] = _T(1);
        __q.memcpy(__d1, __host.data(), __n * sizeof(_T)).wait();

        // Forward tag: the first match.
        EXPECT_TRUE(oneapi::dpl::find_if(__exec_find_if, __d1, __d1 + __n, __is_one) == __d1 + __pos,
                    "wrong index from find_if over 2-byte elements");
        // Or tag: presence only.
        EXPECT_TRUE(oneapi::dpl::any_of(__exec_any_of, __d1, __d1 + __n, __is_one) == __has_match,
                    "wrong result from any_of over 2-byte elements");
        EXPECT_TRUE(oneapi::dpl::none_of(__exec_none_of, __d1, __d1 + __n, __is_one) == !__has_match,
                    "wrong result from none_of over 2-byte elements");
        // Two ranges, so one iteration loads from two streams at 2 bytes each.
        EXPECT_TRUE(oneapi::dpl::mismatch(__exec_mismatch, __d1, __d1 + __n, __d2).first ==
                        (__has_match ? __d1 + __pos : __d1 + __n),
                    "wrong index from mismatch of two 2-byte ranges");
        EXPECT_TRUE(oneapi::dpl::equal(__exec_equal, __d1, __d1 + __n, __d2) == !__has_match,
                    "wrong result from equal of two 2-byte ranges");
    }
    sycl::free(__d1, __q);
    sycl::free(__d2, __q);
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
    {
        test_at_size(__policy, __n);
        test_two_8byte_ranges(__policy, __n);
        test_2byte_ranges(__policy, __n);
    }
#endif
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
