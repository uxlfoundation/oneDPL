// -*- C++ -*-
//===-- batched_stable_sort.cpp -------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../support/test_config.h"

#include <oneapi/dpl/experimental/kernel_templates>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include "radix_sort_utils.h"

namespace kt = oneapi::dpl::experimental::kt;

#ifndef TEST_VALUE_TYPE
#    define TEST_KEYS_ONLY 1
using ValueT = std::uint32_t; // unused by the sorts, only by the reference
#else
using ValueT = TEST_VALUE_TYPE;
#endif
using KeyT = TEST_KEY_TYPE;

using SortConfig = kt::gpu::radix_many_groups_tag<TestRadixBits>;

#include "batched_sort_test_utils.h"

template <bool UsesTempStorage>
struct BatchedStableSort
{
    static constexpr bool uses_temp_storage = UsesTempStorage;

    template <bool IsAscending, typename... Args>
    static sycl::event
    sort(sycl::queue q, Args&&... args)
    {
        return kt::gpu::batched_stable_sort<SortConfig, IsAscending>(q, std::forward<Args>(args)...);
    }
    template <bool IsAscending, typename... Args>
    static sycl::event
    sort_by_key(sycl::queue q, Args&&... args)
    {
        return kt::gpu::batched_stable_sort_by_key<SortConfig, IsAscending>(q, std::forward<Args>(args)...);
    }
    template <typename KernelParam>
    static constexpr std::size_t
    bytes_req(std::size_t segment_size, std::size_t n, bool in_place)
    {
#if TEST_KEYS_ONLY
        return kt::gpu::batched_stable_sort_bytes_req<SortConfig, KeyT>(segment_size, n, in_place, KernelParam{});
#else
        return kt::gpu::batched_stable_sort_by_key_bytes_req<SortConfig, KeyT, ValueT>(segment_size, n, in_place,
                                                                                       KernelParam{});
#endif
    }
};

// Segment sizes around the tile size (data_per_workitem * workgroup_size) and the global histogram chunk (4096), plus
// segments smaller than a tile, multi-tile segments and segments that are not multiples of the sub-group size
std::vector<std::size_t>
segment_sizes(std::size_t tile)
{
    std::vector<std::size_t> sizes = {
        1,        2,      7,        100,      317,          1000,           4095,         4097,
        tile - 1, tile,   tile + 1, 2 * tile, 2 * tile + 1, 3 * tile + 333, 7 * tile - 5, 50'000,
        1 << 17,  300'007};
    sizes.erase(std::remove(sizes.begin(), sizes.end(), std::size_t(0)), sizes.end());
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

int
main()
{
    bool run_test = false;
#if TEST_SYCL_RADIX_SORT_KT_AVAILABLE
    using Param = kt::kernel_param<TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE>;
    using Sorter = BatchedStableSort</*UsesTempStorage=*/false>;
    using SorterTempStorage = BatchedStableSort</*UsesTempStorage=*/true>;
    constexpr Param params;
    // The memory requirement can be known at compile time
    static_assert(SorterTempStorage::bytes_req<Param>(4096, 1 << 20, false) > 0);
    static_assert(SorterTempStorage::bytes_req<Param>(4096, 0, true) == 0);
    auto q = TestUtils::get_test_queue();
#    if TEST_KEYS_ONLY
    run_test = can_run_test<Param, KeyT>(q, params);
#    else
    run_test = can_run_test<Param, KeyT, ValueT>(q, params);
#    endif

    if (run_test)
    {
        try
        {
            const std::size_t tile = std::size_t(Param::data_per_workitem) * Param::workgroup_size;
            const DataMode modes[] = {DataMode::usm_iterators, DataMode::usm_ranges, DataMode::buffer_iterators,
                                      DataMode::buffer_ranges};
            std::size_t mode_idx = 0;
            for (std::size_t segment_size : segment_sizes(tile))
            {
                // A single segment (segment_size == n), a few segments, and many segments. The count of the
                // smallest segments is capped: global scratch memory grows with the number of tiles and segments.
                const std::size_t many = std::clamp<std::size_t>((1 << 20) / segment_size, 2, 2048);
                for (std::size_t segment_count : {std::size_t(1), std::size_t(3), many})
                {
                    const DataMode mode = modes[mode_idx++ % 4];
                    test_case<Sorter, Ascending, false>(q, mode, segment_size, segment_count,
                                                        TestUtils::create_new_kernel_param_idx<0>(params));
                    test_case<Sorter, Descending, false>(q, mode, segment_size, segment_count,
                                                         TestUtils::create_new_kernel_param_idx<1>(params));
                    test_case<Sorter, Ascending, true>(q, mode, segment_size, segment_count,
                                                       TestUtils::create_new_kernel_param_idx<2>(params));
                    test_case<Sorter, Descending, true>(q, mode, segment_size, segment_count,
                                                        TestUtils::create_new_kernel_param_idx<3>(params));
                }

                // Only the segment counts advance the rotation, so each count cycles through all modes
                const DataMode mode = modes[mode_idx % 4];
                test_case<Sorter, Ascending, false>(q, mode, segment_size, many,
                                                    TestUtils::create_new_kernel_param_idx<0>(params),
                                                    DataPattern::all_equal);
                test_case<Sorter, Descending, true>(q, mode, segment_size, many,
                                                    TestUtils::create_new_kernel_param_idx<3>(params),
                                                    DataPattern::reversed);
                test_special_patterns<Sorter>(q, segment_size, params);

                // Temporary storage provided by the caller, for each data mode with in-place and out-of-place sorts
                test_case<SorterTempStorage, Ascending, false>(q, modes[mode_idx % 4], segment_size, 3,
                                                               TestUtils::create_new_kernel_param_idx<0>(params));
                test_case<SorterTempStorage, Descending, true>(q, modes[(mode_idx + 1) % 4], segment_size, 3,
                                                               TestUtils::create_new_kernel_param_idx<3>(params));
            }

            // More histogram chunks than histogram work-groups, with segments of 3 chunks: work-groups own several
            // chunks and switch segments in the middle of their range
            test_case<Sorter, Ascending, false>(q, DataMode::usm_iterators, 9000, 600,
                                                TestUtils::create_new_kernel_param_idx<0>(params));

            test_empty_input<Sorter>(q, params);
            test_empty_input<SorterTempStorage>(q, params, static_cast<void*>(nullptr), std::size_t{0});

            // The kernel_param may be given as a template argument instead, also with temporary storage
            KeyT* no_keys = nullptr;
            kt::gpu::batched_stable_sort<SortConfig, Ascending, Param>(q, no_keys, no_keys, 4).wait();
            kt::gpu::batched_stable_sort<SortConfig, Ascending, Param>(q, no_keys, no_keys, 4, nullptr, 0).wait();
        }
        catch (const std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
    }
#endif // TEST_SYCL_RADIX_SORT_KT_AVAILABLE

    return TestUtils::done(run_test);
}
