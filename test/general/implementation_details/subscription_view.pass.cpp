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

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>
#endif

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    using IntVector = std::vector<int>;

    // Check that __get_subscription_view on IntVector produces the same type
    static_assert(std::is_same_v<IntVector,
                                 std::decay_t<decltype(oneapi::dpl::__ranges::__get_subscription_view(std::declval<IntVector>()))>>);

    // Check that __get_subscription_view is idempotent for std::vector<int>
    static_assert(std::is_same_v<std::decay_t<decltype(oneapi::dpl::__ranges::__get_subscription_view(std::declval<IntVector>()))>,
                                 std::decay_t<decltype(oneapi::dpl::__ranges::__get_subscription_view(
                                                           oneapi::dpl::__ranges::__get_subscription_view(std::declval<IntVector>())))>>);

#if _ENABLE_STD_RANGES_TESTING

    using IteratorOfIntVector = typename IntVector::iterator;
    using MinimalisticRangeForIntVec = TestUtils::MinimalisticView<IteratorOfIntVector>;

    // Check that MinimalisticRangeForIntVec satisfies range, sized_range and view concepts
    static_assert(std::ranges::range      <MinimalisticRangeForIntVec>);
    static_assert(std::ranges::sized_range<MinimalisticRangeForIntVec>);
    static_assert(std::ranges::view       <MinimalisticRangeForIntVec>);

    // Check that __get_subscription_view produces a range, sized_range and view
    using __get_subscription_view_result_t = decltype(oneapi::dpl::__ranges::__get_subscription_view(std::declval<MinimalisticRangeForIntVec>()));
    static_assert(std::ranges::range      <__get_subscription_view_result_t>);
    static_assert(std::ranges::sized_range<__get_subscription_view_result_t>);
    static_assert(std::ranges::view       <__get_subscription_view_result_t>);


    // Check that __get_subscription_view is idempotent for std::vector<int>
    static_assert(std::is_same_v<std::decay_t<__get_subscription_view_result_t>,
                             std::decay_t<decltype(oneapi::dpl::__ranges::__get_subscription_view(
                                                        oneapi::dpl::__ranges::__get_subscription_view(std::declval<MinimalisticRangeForIntVec>())))>>);

    // Check all forms of begin() function
    static_assert(std::is_same_v< decltype(begin(std::declval<      MinimalisticRangeForIntVec>  ())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(begin(std::declval<      MinimalisticRangeForIntVec& >())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(begin(std::declval<const MinimalisticRangeForIntVec& >())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(begin(std::declval<      MinimalisticRangeForIntVec&&>())), IteratorOfIntVector>);
    
    // Check all forms of end() function
    static_assert(std::is_same_v< decltype(end  (std::declval<      MinimalisticRangeForIntVec  >())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(end  (std::declval<      MinimalisticRangeForIntVec& >())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(end  (std::declval<const MinimalisticRangeForIntVec& >())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(end  (std::declval<      MinimalisticRangeForIntVec&&>())), IteratorOfIntVector>);
    
    // Check that MinimalisticView with vector<int>::iterator is a range
    static_assert(std::ranges::range<TestUtils::MinimalisticView<IntVector::iterator>>);

    // All oneDPL algorithms require at least a random access range
    static_assert(std::ranges::random_access_range<TestUtils::MinimalisticView<IntVector::iterator>>);

#endif // _ENABLE_STD_RANGES_TESTING
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
