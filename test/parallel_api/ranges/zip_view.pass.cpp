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

#include "support/test_config.h"

#include <oneapi/dpl/execution>

#if _ENABLE_RANGES_TESTING
#    include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"
#include "support/utils_device_copyable.h"

#include <forward_list>
#include <iostream>

std::int32_t
main()
{
    bool run = false;

#if _ENABLE_RANGES_TESTING

    using namespace oneapi::dpl::experimental::ranges;

#if !(_ONEDPL_CPP20_RANGES_PRESENT && TEST_STD_RANGES_VIEW_CONCEPT_REQUIRES_DEFAULT_INITIALIZABLE)
    run = true;
    {
        constexpr int max_n = 10;
        char data[max_n] = {'b', 'e', 'g', 'f', 'c', 'd', 'a', 'j', 'i', 'h'};
        int key[max_n] = {1, 4, 6, 5, 2, 3, 0, 9, 8, 7};

        //the name nano::ranges::views::all is not injected into oneapi::dpl::experimental::ranges namespace
        auto view = __nanorange::nano::views::all(data);
#if _ONEDPL_CPP20_OWNING_VIEW_PRESENT
        auto z = zip_view(view, __nanorange::nano::views::all(key));
#else
        auto key_view = __nanorange::nano::views::all(key);
        auto z = zip_view(view, key_view);
#endif
        //check access
        EXPECT_TRUE(std::get<0>(z[2]) == 'g', "wrong effect with zip_view");

        int64_t max_int32p2 = (size_t)std::numeric_limits<int32_t>::max() + 2L;

        auto base_view = views::iota(std::int64_t(0), max_int32p2);

        //avoiding allocating large amounts of memory, just reusing small data container
        auto transform_data_idx = [&max_n, &data](auto idx) { return data[idx % max_n]; };
        auto data_large_view = views::transform(base_view, transform_data_idx);

        //avoiding allocating large amounts of memory, just reusing small data container
        auto transform_key_idx = [&max_n, &key](auto idx) { return key[idx % max_n]; };
        auto key_large_view = views::transform(base_view, transform_key_idx);

        auto large_z = zip_view(data_large_view, key_large_view);

        //check that zip_view ranges can be larger than a signed 32 bit integer
        size_t i = large_z.size() - 1;

        auto expected_key = key[i % max_n];
        auto actual_key = std::get<1>(large_z[i]);
        EXPECT_EQ(expected_key, actual_key, "wrong effect with zip_view bracket operator");

        char expected_data = data[i % max_n];
        char actual_data = std::get<0>(large_z[i]);
        EXPECT_EQ(expected_data, actual_data, "wrong effect with zip_view bracket operator");
    }
#endif // !(_ONEDPL_CPP20_RANGES_PRESENT && TEST_STD_RANGES_VIEW_CONCEPT_REQUIRES_DEFAULT_INITIALIZABLE)

#if _ONEDPL_CPP20_RANGES_PRESENT
    run = true;
    {
        //check basic zip_view construction and access with std C++20 views
        auto v1 = std::views::iota(0, 5);
        auto v2 = std::views::iota(10, 15);
        auto z = zip_view(v1, v2);

        EXPECT_TRUE(std::get<0>(z[0]) == 0, "wrong effect with zip_view (std ranges)");
        EXPECT_TRUE(std::get<1>(z[0]) == 10, "wrong effect with zip_view (std ranges)");
        EXPECT_TRUE(std::get<0>(z[3]) == 3, "wrong effect with zip_view (std ranges)");
        EXPECT_TRUE(std::get<1>(z[3]) == 13, "wrong effect with zip_view (std ranges)");
        EXPECT_EQ(5u, z.size(), "wrong size with zip_view (std ranges)");
    }
    {
        //check iterator-sentinel equality using an unbounded iota view, which produces
        //a sentinel type distinct from its iterator type
        auto unbounded = std::views::iota(0);
        auto z = zip_view(unbounded);
        auto it = z.begin();
        auto sent = z.end();

        // iterator == sentinel (member operator==)
        EXPECT_TRUE(!(it == sent), "iterator should not equal sentinel for unbounded range");
        // sentinel == iterator (C++20 rewritten candidate)
        EXPECT_TRUE(!(sent == it), "sentinel should not equal iterator for unbounded range");
        // also verify != in both directions
        EXPECT_TRUE(it != sent, "iterator != sentinel failed for unbounded range");
        EXPECT_TRUE(sent != it, "sentinel != iterator failed for unbounded range");
    }
    {
        //check iterator-sentinel operator- using a forward-only range with a sized sentinel.
        //std::counted_iterator + std::default_sentinel satisfies sized_sentinel_for but is
        //forward-only, which prevents __zip_is_common from collapsing end() to an iterator.
        std::forward_list<int> fl = {1, 2, 3, 4, 5};
        auto counted = std::ranges::subrange(std::counted_iterator(fl.begin(), 5), std::default_sentinel);
        auto z = zip_view(counted);
        auto it = z.begin();
        auto sent = z.end();

        // iterator - sentinel (member operator-)
        EXPECT_EQ(-5, it - sent, "wrong result for zip_view iterator - sentinel");
        // sentinel - iterator (hidden friend, delegates to member)
        EXPECT_EQ(5, sent - it, "wrong result for zip_view sentinel - iterator");
    }
#endif // _ONEDPL_CPP20_RANGES_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
    run = true;
    {
        // zip_view: device copyable if all ranges are device copyable
        static_assert(
            sycl::is_device_copyable_v<
                zip_view<TestUtils::range_device_copyable, TestUtils::range_device_copyable>>,
            "zip_view is not device copyable with device copyable ranges");

        static_assert(!sycl::is_device_copyable_v<zip_view<TestUtils::range_non_device_copyable,
                                                           TestUtils::range_device_copyable>>,
                      "zip_view is device copyable with non device copyable ranges");

        static_assert(
            !sycl::is_device_copyable_v<zip_view<TestUtils::range_device_copyable,
                                                 TestUtils::range_non_device_copyable>>,
            "zip_view is device copyable with non device copyable ranges");
    }

#endif     // TEST_DPCPP_BACKEND_PRESENT
#endif // _ENABLE_RANGES_TESTING

    return TestUtils::done(run);
}
