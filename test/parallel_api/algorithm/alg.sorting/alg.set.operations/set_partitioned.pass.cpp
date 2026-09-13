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

// Covers the partitioned merge path of the hetero set operations, which the four set_<op>.pass tests never
// reach: it is gated behind _ONEDPL_SET_OP_PARTITION_THRESHOLD total input elements, which is above every
// size those tests use. test/CMakeLists.txt builds this source three ways:
//
//   set_partitioned.pass                  default gate and device-derived tile: inputs must exceed 2M
//                                         elements, so this is the only variant covering the shipped
//                                         configuration
//   set_partitioned_low_threshold.pass    threshold lowered, tile still device-derived
//   set_partitioned_dense_tiles.pass      threshold lowered and the tile forced to three diagonals
//
// The derived tile scales with the device's local memory and with the input value types -- 256 diagonals on
// a 64 KiB device with int32, 512 on 128 KiB -- so how many tile boundaries a given input crosses is a
// property of the device, not of the test. Forcing the tile is what makes that count fixed and small.
//
// The path is SYCL-only, so every variant uses invoke_on_all_hetero_policies; the host policies would only
// re-run what set_<op>.pass already covers, and at the size the default variant needs they are what makes
// the test exceed oneDPL's CI per-test timeout.

#ifndef _ONEDPL_TEST_SET_OP_LOW_THRESHOLD
#    define _ONEDPL_TEST_SET_OP_LOW_THRESHOLD 0
#endif
#ifndef _ONEDPL_TEST_SET_OP_DENSE_TILES
#    define _ONEDPL_TEST_SET_OP_DENSE_TILES 0
#endif

// A dense-tile build implies the lowered threshold, since a tile override is meaningless at an input size
// the suite cannot reach. Both library macros must be set before oneDPL is included.
#if _ONEDPL_TEST_SET_OP_DENSE_TILES
#    define _ONEDPL_SET_OP_PARTITION_THRESHOLD 1024
#    define _ONEDPL_SET_OP_PARTITION_TILE_DIAGONALS 3
#elif _ONEDPL_TEST_SET_OP_LOW_THRESHOLD
#    define _ONEDPL_SET_OP_PARTITION_THRESHOLD 1024
#endif

#include "set_common.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/utils_const.h" // get_scan_test_set_max_n

#    include <cstdlib>
#    include <iostream>
#    include <utility>

using SizePair = std::pair<std::size_t, std::size_t>;

constexpr bool kThresholdLowered = _ONEDPL_TEST_SET_OP_DENSE_TILES || _ONEDPL_TEST_SET_OP_LOW_THRESHOLD;

// Seeded per (size pair, operation) rather than once per run, so that the data a failing case saw does not
// depend on which cases ran before it. get_scan_test_set_max_n() drops the larger pairs on a CPU device, so
// a single stream would make a GPU failure unreproducible on any other device.
template <template <typename> class TestType, typename ValueT>
void
test_set_sizes(std::size_t n1, std::size_t n2, unsigned seed)
{
    std::srand(seed);

    // A value range that grows more slowly than the index produces long runs of duplicates, which is what
    // drives the balanced path star correction of the hetero partitioned merge path.
    Sequence<ValueT> in1(n1, [](std::size_t k) { return rand() % std::max(std::size_t{3}, k >> 4); });
    Sequence<ValueT> in2(n2, [](std::size_t k) { return rand() % std::max(std::size_t{3}, k >> 4); });

    std::sort(in1.begin(), in1.end());
    std::sort(in2.begin(), in2.end());

    invoke_on_all_hetero_policies<0>()(TestType<ValueT>(), in1.begin(), in1.end(), in2.cbegin(), in2.cend(),
                                       oneapi::dpl::__internal::__pstl_less());
}

// Reports the configuration in effect, which is otherwise invisible in a test log: the derived tile size
// depends on the device's local memory, so the number of tiles a given input covers is not knowable from
// the source alone.
void
announce_config()
{
    const auto local_mem_size = TestUtils::get_test_queue().get_device().get_info<sycl::info::device::local_mem_size>();
    std::cout << "set_partitioned: threshold=" << std::size_t{_ONEDPL_SET_OP_PARTITION_THRESHOLD}
              << " tile_diagonals_override=" << std::size_t{_ONEDPL_SET_OP_PARTITION_TILE_DIAGONALS}
              << " local_mem_size=" << local_mem_size << " is_cpu=" << TestUtils::test_queue_is_cpu() << std::endl;
}

// Runs all four set operations over each size pair. Each pair is announced before it runs, so that a
// failure reported only as an index into the output can be attributed to a size pair; the failure message
// itself names the operation. The seed is derived from the size pair, not from the loop index, so it is the
// same whether or not earlier pairs were filtered out.
void
run_size_pairs(const SizePair* sizes, std::size_t num_sizes)
{
    using ValueT = std::int32_t;

    for (std::size_t i = 0; i != num_sizes; ++i)
    {
        const std::size_t n1 = sizes[i].first;
        const std::size_t n2 = sizes[i].second;
        const unsigned seed = static_cast<unsigned>(4200 + n1 * 31 + n2);
        std::cout << "set_partitioned: n1=" << n1 << " n2=" << n2 << " total=" << n1 + n2 << " seed=" << seed
                  << std::endl;

        test_set_sizes<test_set_union, ValueT>(n1, n2, seed);
        test_set_sizes<test_set_intersection, ValueT>(n1, n2, seed + 1);
        test_set_sizes<test_set_difference, ValueT>(n1, n2, seed + 2);
        test_set_sizes<test_set_symmetric_difference, ValueT>(n1, n2, seed + 3);
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
    bool ran = false;
#if TEST_DPCPP_BACKEND_PRESENT && !ONEDPL_FPGA_DEVICE
    announce_config();

    if (kThresholdLowered)
    {
        const SizePair sizes[] = {
            {511, 512},       // below the lowered threshold, so the unpartitioned fallback
            {512, 512},       // the smallest partitioned input
            {1000, 24},       // at the threshold, strongly asymmetric
            {8192, 8192},     //
            {8193, 8192},     // one diagonal past a power of two
            {20000, 5000},    // asymmetric
            {50000, 50000},   //
            {100000, 100000}, // the largest input the suite uses
            {100000, 3}       // extreme asymmetry: the rng1 tail drain dominates
        };
        const std::size_t n_max = TestUtils::get_scan_test_set_max_n();

        SizePair in_range[std::size(sizes)];
        std::size_t num_in_range = 0;
        for (const auto& s : sizes)
        {
            if (std::max(s.first, s.second) <= n_max)
                in_range[num_in_range++] = s;
        }
        run_size_pairs(in_range, num_in_range);
        ran = true;
    }
    // The default threshold needs more than 2M input elements, far above get_scan_test_set_max_n(), so that
    // cap cannot be honoured here: no size within it reaches this path at all. Cost is contained by running
    // hetero policies only -- the host ones are what makes this size expensive, and they add no coverage of
    // a SYCL-only path -- and by one size pair instead of two in debug, where the serial reference pass over
    // 2.2M elements dominates.
    else
    {
        const SizePair sizes[] = {
            {1100000, 1100000}, // 2.2M total: just past the threshold, so the tile count is small
            {2199984, 16}       // the same total, extremely asymmetric
        };
#    if PSTL_USE_DEBUG
        run_size_pairs(sizes, 1);
#    else
        run_size_pairs(sizes, std::size(sizes));
#    endif
        ran = true;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT && !ONEDPL_FPGA_DEVICE

    return TestUtils::done(ran);
}
