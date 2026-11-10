// -*- C++ -*-
//===-- radix_sort_residency.cpp ------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that the cooperative onesweep kernel does not hang when the input has more tiles than work-groups which can
// be resident at once. Each work-group processes tiles in a strided loop and the lookback of a tile spins on its
// predecessor tile. If too many work-groups are launched, the sort deadlocks once every resident work-group holds a
// tile whose predecessor belongs to a non-resident work-group. Twice the upper bound on resident work-groups
// guarantees this for any over-estimated work-group count.
//
// A key-value sort is tested if TEST_VALUE_TYPE is defined, otherwise a key sort.

#include "../support/test_config.h"

#include <oneapi/dpl/experimental/kernel_templates>
#include <oneapi/dpl/iterator>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include "../support/utils.h"
#include "../support/sycl_alloc_utils.h"

#include "radix_sort_utils.h"

#if TEST_SYCL_RADIX_SORT_KT_AVAILABLE

constexpr std::size_t elems_in_tile = std::size_t{TEST_DATA_PER_WORK_ITEM} * TEST_WORK_GROUP_SIZE;

std::size_t
get_residency_saturating_size(const sycl::queue& q)
{
    // Upper bound on resident work-groups which ignores SLM limits
    constexpr std::size_t lanes_per_xe = 2048;
    constexpr std::size_t xve_per_xe = 8;
    const std::size_t xes_on_device = q.get_device().get_info<sycl::info::device::max_compute_units>() / xve_per_xe;
    const std::size_t residency_bound = (lanes_per_xe / TEST_WORK_GROUP_SIZE) * xes_on_device;
    // The extra element covers the partial tile path.
    return 2 * residency_bound * elems_in_tile + 1;
}

template <typename KernelParam>
void
test_device_usm(sycl::queue q, std::size_t size, KernelParam param)
{
    std::vector<TEST_KEY_TYPE> expected_keys(size);
    TestUtils::generate_arithmetic_data(expected_keys.data(), size, 6);
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, TEST_KEY_TYPE> keys(q, expected_keys.begin(),
                                                                                expected_keys.end());
    const std::string msg_suffix = ", n: " + std::to_string(size);
#    ifdef TEST_VALUE_TYPE
    std::vector<TEST_VALUE_TYPE> expected_values(size);
    TestUtils::generate_arithmetic_data(expected_values.data(), size, 7);
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, TEST_VALUE_TYPE> values(q, expected_values.begin(),
                                                                                    expected_values.end());

    kt_ns::radix_sort_by_key<Ascending, TestRadixBits>(q, keys.get_data(), keys.get_data() + size, values.get_data(),
                                                       param)
        .wait();

    auto expected_first = oneapi::dpl::make_zip_iterator(expected_keys.begin(), expected_values.begin());
    std::stable_sort(expected_first, expected_first + size, CompareKey<Ascending>{});

    std::vector<TEST_VALUE_TYPE> actual_values(size);
    values.retrieve_data(actual_values.begin());
    EXPECT_EQ_N(expected_values.begin(), actual_values.begin(), size, ("wrong values" + msg_suffix).c_str());
#    else
    kt_ns::radix_sort<Ascending, TestRadixBits>(q, keys.get_data(), keys.get_data() + size, param).wait();

    std::stable_sort(expected_keys.begin(), expected_keys.end(), Compare<TEST_KEY_TYPE, Ascending>{});
#    endif
    std::vector<TEST_KEY_TYPE> actual_keys(size);
    keys.retrieve_data(actual_keys.begin());
    EXPECT_EQ_N(expected_keys.begin(), actual_keys.begin(), size, ("wrong keys" + msg_suffix).c_str());
}

#endif // TEST_SYCL_RADIX_SORT_KT_AVAILABLE

int
main()
{
    bool run_test = false;
#if TEST_SYCL_RADIX_SORT_KT_AVAILABLE
    constexpr oneapi::dpl::experimental::kt::kernel_param<TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE> params;
    auto q = TestUtils::get_test_queue();
#    ifdef TEST_VALUE_TYPE
    run_test = can_run_test<decltype(params), TEST_KEY_TYPE, TEST_VALUE_TYPE>(q, params);
#    else
    run_test = can_run_test<decltype(params), TEST_KEY_TYPE>(q, params);
#    endif
    if (run_test)
    {
        try
        {
            // Test a case with a small number of tiles in addition to exceeding max residency
            test_device_usm(q, 2 * elems_in_tile + 1, params);
            test_device_usm(q, get_residency_saturating_size(q), params);
        }
        catch (const std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
    }
#endif

    return TestUtils::done(run_test);
}
