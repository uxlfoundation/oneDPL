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

#if _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT
#include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>

using IntVector = std::vector<int>;
using IteratorOfIntVector = typename IntVector::iterator;

using MinimalisticRangeForIntVec     = TestUtils::MinimalisticRange<IteratorOfIntVector>;
using MinimalisticRangeViewForIntVec = TestUtils::MinimalisticView <IteratorOfIntVector>;

template <typename RandomIt>
struct MinimalisticViewWithSubscription : TestUtils::MinimalisticView<RandomIt>
{
    MinimalisticViewWithSubscription(RandomIt it_begin, RandomIt it_end)
        : TestUtils::MinimalisticView<RandomIt>(it_begin, it_end)
    {
    }

    auto operator[](std::size_t index) const
    {
        return *(oneapi::dpl::__ranges::__begin(*this) + index);
    }
};

template <typename _Rng>
inline constexpr bool contains_host_pointer_v = oneapi::dpl::__ranges::__contains_host_pointer<_Rng>::value;

template <typename _Rng>
inline constexpr bool contains_host_pointer_on_any_layers_v = oneapi::dpl::__ranges::__contains_host_pointer_on_any_layers_v<_Rng>;

// oneapi::dpl::__ranges::__contains_host_pointer functional
void
check_contains_host_pointer()
{
    // Check that MinimalisticRangeForIntVec can't be used in oneDPL algorithms
    // as far it contains host pointers in the std::ranges::ref_view
    {
        IntVector vec;
        MinimalisticRangeForIntVec mr(vec.begin(), vec.end());
        auto all_view = std::ranges::views::all(mr);
        static_assert(contains_host_pointer_v<decltype(all_view)> == true);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(all_view)> == true);
    }

    // Check that MinimalisticRangeForIntVec can be used in oneDPL algorithms
    // as far it doesn't contains host pointers in the std::ranges::ref_view in this case
    {
        IntVector vec;
        auto all_view = std::ranges::views::all(MinimalisticRangeForIntVec(vec.begin(), vec.end()));
        static_assert(contains_host_pointer_v<decltype(all_view)> == false);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(all_view)> == false);
    }

    // Check that MinimalisticViewWithSubscription can be used in oneDPL algorithms as far it doesn't contains host pointers
    {
        IntVector vec;
        MinimalisticViewWithSubscription mr_view(vec.begin(), vec.end());
        auto all_view = std::ranges::views::all(mr_view);
        static_assert(contains_host_pointer_v<decltype(all_view)> == false);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(all_view)> == false);
    }

    // Check that std::string_view can't be used in oneDPL algorithms
    // as far it contains host pointers
    {
        std::string str;
        std::string_view str_view(str);
        static_assert(contains_host_pointer_v<decltype(str_view)> == true);
    }

#if TEST_CPP20_SPAN_PRESENT
    // Check that std::span can't be used in oneDPL algorithms
    // as far it contains host pointers
    {
        int int_array[5] = {1, 2, 3, 4, 5};
        std::span span_view = int_array;
        static_assert(contains_host_pointer_v<decltype(span_view)> == true);
    }
#endif
}

// oneapi::dpl::__ranges::__contains_host_pointer functional with oneapi::dpl::__ranges::zip_view
void
check_contains_host_pointer_in_zip_view()
{
    // Check that MinimalisticRangeForIntVec can't be used in oneDPL algorithms
    // as far it contains host pointers in the std::ranges::ref_view
    {
        IntVector vec;
        MinimalisticRangeForIntVec mr(vec.begin(), vec.end());
        auto all_view = std::ranges::views::all(mr);
        auto zip_view = oneapi::dpl::__ranges::make_zip_view(all_view, all_view);

        static_assert(contains_host_pointer_v<decltype(all_view)> == true);
        static_assert(contains_host_pointer_v<decltype(zip_view)> == false);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(zip_view)> == true);
    }

    // Check that MinimalisticViewWithSubscription can be used in oneDPL algorithms
    // as far it contains host pointers in the std::ranges::ref_view
    {
        IntVector vec;
        MinimalisticViewWithSubscription mrv(vec.begin(), vec.end());
        auto all_view = std::ranges::views::all(mrv);
        auto zip_view = oneapi::dpl::__ranges::make_zip_view(all_view, all_view);

        static_assert(contains_host_pointer_v<decltype(all_view)> == false);
        static_assert(contains_host_pointer_v<decltype(zip_view)> == false);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(zip_view)> == false);
    }
}

// oneapi::dpl::__ranges::__contains_host_pointer functional with std::ranges::take_view
void
check_contains_host_pointer_in_take_view()
{
    // Check that MinimalisticRangeForIntVec can't be used in oneDPL algorithms
    // as far it contains host pointers in the std::ranges::ref_view
    IntVector vec;
    MinimalisticRangeForIntVec mr(vec.begin(), vec.end());
    auto all_view = std::ranges::views::all(mr);
    auto taken_view = std::ranges::take_view(all_view, all_view.size());

    static_assert(contains_host_pointer_v<decltype(all_view)> == true);
    static_assert(contains_host_pointer_v<decltype(taken_view)> == false);
    static_assert(contains_host_pointer_on_any_layers_v<decltype(taken_view)> == true);
}

// oneapi::dpl::__ranges::__contains_host_pointer functional with std::ranges::drop_view
void
check_contains_host_pointer_in_drop_view()
{
    // Check that MinimalisticRangeForIntVec can't be used in oneDPL algorithms
    // as far it contains host pointers in the std::ranges::ref_view
    IntVector vec;
    MinimalisticRangeForIntVec mr(vec.begin(), vec.end());
    auto all_view = std::ranges::views::all(mr);
    auto dropped_view = std::ranges::drop_view(all_view, 0);

    static_assert(contains_host_pointer_v<decltype(all_view)> == true);
    static_assert(contains_host_pointer_v<decltype(dropped_view)> == false);
    static_assert(contains_host_pointer_on_any_layers_v<decltype(dropped_view)> == true);
}
#endif // _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT

int
main()
{
    bool bProcessed = false;

#if _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT

    // Check that __get_subscription_view on IntVector produces the same type
    static_assert(std::is_same_v<IntVector,
                                 std::decay_t<decltype(oneapi::dpl::__ranges::__get_subscription_view(std::declval<IntVector>()))>>);

    // Check that __get_subscription_view is idempotent for std::vector<int>
    static_assert(std::is_same_v<std::decay_t<decltype(oneapi::dpl::__ranges::__get_subscription_view(std::declval<IntVector>()))>,
                                 std::decay_t<decltype(oneapi::dpl::__ranges::__get_subscription_view(
                                                           oneapi::dpl::__ranges::__get_subscription_view(std::declval<IntVector>())))>>);

    check_contains_host_pointer();
    check_contains_host_pointer_in_zip_view();
    check_contains_host_pointer_in_take_view();
    check_contains_host_pointer_in_drop_view();

    // Check that MinimalisticRangeViewForIntVec satisfies range, sized_range and view concepts
    static_assert(std::ranges::range      <MinimalisticRangeViewForIntVec>);
    static_assert(std::ranges::sized_range<MinimalisticRangeViewForIntVec>);
    static_assert(std::ranges::view       <MinimalisticRangeViewForIntVec>);

    // Check that __get_subscription_view produces a range, sized_range and view
    using __get_subscription_view_result_t = decltype(oneapi::dpl::__ranges::__get_subscription_view(std::declval<MinimalisticRangeViewForIntVec>()));
    static_assert(std::ranges::range      <__get_subscription_view_result_t>);
    static_assert(std::ranges::sized_range<__get_subscription_view_result_t>);
    static_assert(std::ranges::view       <__get_subscription_view_result_t>);


    // Check that __get_subscription_view is idempotent for std::vector<int>
    static_assert(std::is_same_v<std::decay_t<__get_subscription_view_result_t>,
                             std::decay_t<decltype(oneapi::dpl::__ranges::__get_subscription_view(
                                                        oneapi::dpl::__ranges::__get_subscription_view(std::declval<MinimalisticRangeViewForIntVec>())))>>);

    // Check all forms of begin() function
    static_assert(std::is_same_v< decltype(begin(std::declval<      MinimalisticRangeViewForIntVec>  ())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(begin(std::declval<      MinimalisticRangeViewForIntVec& >())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(begin(std::declval<const MinimalisticRangeViewForIntVec& >())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(begin(std::declval<      MinimalisticRangeViewForIntVec&&>())), IteratorOfIntVector>);
    
    // Check all forms of end() function
    static_assert(std::is_same_v< decltype(end  (std::declval<      MinimalisticRangeViewForIntVec  >())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(end  (std::declval<      MinimalisticRangeViewForIntVec& >())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(end  (std::declval<const MinimalisticRangeViewForIntVec& >())), IteratorOfIntVector>);
    static_assert(std::is_same_v< decltype(end  (std::declval<      MinimalisticRangeViewForIntVec&&>())), IteratorOfIntVector>);
    
    // Check that MinimalisticView with vector<int>::iterator is a range
    static_assert(std::ranges::range<TestUtils::MinimalisticView<IntVector::iterator>>);

    // All oneDPL algorithms require at least a random access range
    static_assert(std::ranges::random_access_range<TestUtils::MinimalisticView<IntVector::iterator>>);

#endif // _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(bProcessed);
}
