// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

// Driver shared by the set_partitioned_* tests, which cover the partitioned merge path of the hetero set
// operations. It lives outside set_common.h so that the four set_<op>.pass tests do not instantiate it.

#ifndef _ONEDPL_SET_PARTITIONED_COMMON_H
#define _ONEDPL_SET_PARTITIONED_COMMON_H

#include "set_common.h"

#include <cstdlib>
#include <iostream>
#include <utility>

using SizePair = std::pair<std::size_t, std::size_t>;

template <template <typename> class TestType, typename ValueT>
void
test_set_sizes(std::size_t n1, std::size_t n2)
{
    // A value range that grows more slowly than the index produces long runs of duplicates, which is what
    // drives the balanced path star correction of the hetero partitioned merge path.
    Sequence<ValueT> in1(n1, [](std::size_t k) { return rand() % std::max(std::size_t{3}, k >> 4); });
    Sequence<ValueT> in2(n2, [](std::size_t k) { return rand() % std::max(std::size_t{3}, k >> 4); });

    ::std::sort(in1.begin(), in1.end());
    ::std::sort(in2.begin(), in2.end());

    invoke_on_all_policies<0>()(TestType<ValueT>(), in1.begin(), in1.end(), in2.cbegin(), in2.cend(),
                                oneapi::dpl::__internal::__pstl_less());
}

// Reports the tile configuration in effect, which is otherwise invisible in a test log: the derived tile
// size depends on the device's local memory, so the number of tiles a given input covers is not knowable
// from the source alone.
inline void
announce_partition_config()
{
    std::cout << "set_partitioned: threshold=" << std::size_t{_ONEDPL_SET_OP_PARTITION_THRESHOLD}
              << " tile_diagonals_override=" << std::size_t{_ONEDPL_SET_OP_PARTITION_TILE_DIAGONALS}
              << " no_last_store=" << int{_ONEDPL_SET_OP_DIAG_NO_LAST_STORE}
              << " full_range_bounds=" << int{_ONEDPL_SET_OP_DIAG_FULL_RANGE_BOUNDS};
#if TEST_DPCPP_BACKEND_PRESENT
    const auto device = TestUtils::get_test_queue().get_device();
    std::cout << " local_mem_size=" << device.get_info<sycl::info::device::local_mem_size>()
              << " is_cpu=" << TestUtils::test_queue_is_cpu()
              << " name='" << device.get_info<sycl::info::device::name>() << "'"
              << " driver='" << device.get_info<sycl::info::device::driver_version>() << "'"
              << " max_wg=" << device.get_info<sycl::info::device::max_work_group_size>() << " sub_groups=";
    for (std::size_t sg : device.get_info<sycl::info::device::sub_group_sizes>())
        std::cout << sg << ",";
#endif
    std::cout << std::endl;
}

// Runs all four set operations over each size pair. Each pair is announced before it runs, so that a
// failure reported only as an index into the output can be attributed to a size pair; the failure message
// itself names the operation.
inline void
run_test_set_partitioned(const SizePair* sizes, std::size_t num_sizes)
{
    using ValueT = std::int32_t;

    announce_partition_config();

    ::std::srand(4200);
    for (std::size_t i = 0; i != num_sizes; ++i)
    {
        const std::size_t n1 = sizes[i].first;
        const std::size_t n2 = sizes[i].second;
        std::cout << "set_partitioned: n1=" << n1 << " n2=" << n2 << " total=" << n1 + n2 << std::endl;

        test_set_sizes<test_set_union, ValueT>(n1, n2);
        test_set_sizes<test_set_intersection, ValueT>(n1, n2);
        test_set_sizes<test_set_difference, ValueT>(n1, n2);
        test_set_sizes<test_set_symmetric_difference, ValueT>(n1, n2);
    }
}

// Size pairs straddling a threshold lowered to 1024. Meaningful only for a test that lowers it, since the
// default is above every size here.
inline void
run_test_set_partitioned_small()
{
    const SizePair sizes[] = {
        {511, 512},       // total 1023: below a threshold lowered to 1024, so the unpartitioned fallback
        {512, 512},       // total 1024: the smallest partitioned input
        {1000, 24},       // at the threshold, strongly asymmetric
        {8192, 8192},     //
        {8193, 8192},     // one diagonal past a power of two
        {20000, 5000},    // asymmetric
        {50000, 50000},   //
        {100000, 100000}, // the largest input the suite uses
        {100000, 3}       // extreme asymmetry: the rng1 tail drain dominates
    };

    const std::size_t n_max = TestUtils::get_scan_test_set_max_n();

    SizePair in_range[sizeof(sizes) / sizeof(sizes[0])];
    std::size_t num_in_range = 0;
    for (const auto& s : sizes)
    {
        if (std::max(s.first, s.second) <= n_max)
            in_range[num_in_range++] = s;
    }

    run_test_set_partitioned(in_range, num_in_range);
}

#endif // _ONEDPL_SET_PARTITIONED_COMMON_H
