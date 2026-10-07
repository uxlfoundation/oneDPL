// -*- C++ -*-
//===-- batched_merge_sort.cpp --------------------------------------------===//
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

namespace kt = oneapi::dpl::experimental::kt;

#ifndef TEST_VALUE_TYPE
#    define TEST_KEYS_ONLY 1
using ValueT = std::uint32_t; // unused by the sorts, only by the reference
#else
using ValueT = TEST_VALUE_TYPE;
#endif
using KeyT = TEST_KEY_TYPE;

constexpr bool Ascending = true;
constexpr bool Descending = false;

#include "batched_sort_test_utils.h"

struct BatchedMergeSort
{
    template <bool IsAscending, typename... Args>
    static sycl::event
    sort(sycl::queue q, Args&&... args)
    {
        return kt::gpu::batched_merge_sort<IsAscending>(q, std::forward<Args>(args)...);
    }
    template <bool IsAscending, typename... Args>
    static sycl::event
    sort_by_key(sycl::queue q, Args&&... args)
    {
        return kt::gpu::batched_merge_sort_by_key<IsAscending>(q, std::forward<Args>(args)...);
    }
};

std::vector<std::size_t>
segment_sizes(std::size_t dpwi, std::size_t capacity)
{
    std::vector<std::size_t> sizes = {1,       2,   3,   7,    dpwi,         dpwi + 1,         2 * dpwi - 1,
                                      100,     256, 317, 1000, capacity / 2, capacity / 2 + 3, capacity - 1,
                                      capacity};
    sizes.erase(std::remove_if(sizes.begin(), sizes.end(), [&](std::size_t s) { return s == 0 || s > capacity; }),
                sizes.end());
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

template <typename KernelParam>
bool
can_run_test(sycl::queue q)
{
    std::size_t elem_bytes = sizeof(KeyT);
#if !TEST_KEYS_ONLY
    elem_bytes += sizeof(ValueT);
#endif
    const std::size_t slm = std::size_t(KernelParam::data_per_workitem) * KernelParam::workgroup_size * elem_bytes;
    const auto device = q.get_device();
    return slm <= device.get_info<sycl::info::device::local_mem_size>() &&
           KernelParam::workgroup_size <= device.get_info<sycl::info::device::max_work_group_size>();
}

int
main()
{
    using Param = kt::kernel_param<TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE>;
    using Sorter = BatchedMergeSort;
    constexpr Param params;
    auto q = TestUtils::get_test_queue();
    const bool run_test = can_run_test<Param>(q);

    if (run_test)
    {
        try
        {
            const std::size_t capacity = std::size_t(Param::data_per_workitem) * Param::workgroup_size;
            const DataMode modes[] = {DataMode::usm_iterators, DataMode::usm_ranges, DataMode::buffer_iterators,
                                      DataMode::buffer_ranges};
            std::size_t mode_idx = 0;
            for (std::size_t segment_size : segment_sizes(Param::data_per_workitem, capacity))
            {
                // Vary the segment count: a single segment, a partially filled last work-group, and many segments
                const std::size_t many = std::max<std::size_t>(3, (1 << 18) / segment_size);
                for (std::size_t segment_count : {std::size_t(1), std::size_t(37), many})
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

                const DataMode mode = modes[mode_idx++ % 4];
                test_case<Sorter, Ascending, false>(q, mode, segment_size, many,
                                                    TestUtils::create_new_kernel_param_idx<0>(params),
                                                    DataPattern::all_equal);
                test_case<Sorter, Descending, true>(q, mode, segment_size, many,
                                                    TestUtils::create_new_kernel_param_idx<3>(params),
                                                    DataPattern::reversed);
                test_special_patterns<Sorter>(q, segment_size, params);
            }

            test_empty_input<Sorter>(q, params);
        }
        catch (const std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    return TestUtils::done(run_test);
}
