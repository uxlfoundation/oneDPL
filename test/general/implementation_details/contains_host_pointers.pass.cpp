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
#if TEST_STD_RANGES_VIEW_CONCEPT_REQUIRES_DEFAULT_INITIALIZABLE
    MinimalisticViewWithSubscription() = default;
#endif

    MinimalisticViewWithSubscription(RandomIt it_begin, RandomIt it_end)
        : TestUtils::MinimalisticView<RandomIt>(it_begin, it_end)
    {
    }

    auto operator[](std::size_t index) const
    {
        return *(oneapi::dpl::__ranges::__begin(*this) + index);
    }
};

template <typename RandomIt>
RandomIt
begin(MinimalisticViewWithSubscription<RandomIt> view)
{
    return view.it_begin;
}

template <typename RandomIt>
RandomIt
end(MinimalisticViewWithSubscription<RandomIt> view)
{
    return view.it_end;
}

template <typename _Rng>
inline constexpr bool contains_host_pointer_v = oneapi::dpl::__ranges::__contains_host_pointer<_Rng>::value;

template <typename _Rng>
inline constexpr bool contains_host_pointer_on_any_layers_v = oneapi::dpl::__ranges::__contains_host_pointer_on_any_layers<_Rng>::value;

// oneapi::dpl::__ranges::__contains_host_pointer functional
void
check_contains_host_pointer()
{
    {
        IntVector vec;
        MinimalisticRangeForIntVec mr(vec.begin(), vec.end());
        auto all_view = std::ranges::views::all(mr);
        static_assert(contains_host_pointer_v<decltype(all_view)> == true);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(all_view)> == true);
    }

#if !TEST_STD_RANGES_VIEWABLE_RANGE_CONCEPT_BROKEN
    {
        IntVector vec;
        auto all_view = std::ranges::views::all(MinimalisticRangeForIntVec(vec.begin(), vec.end()));
        static_assert(contains_host_pointer_v<decltype(all_view)> == false);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(all_view)> == false);
    }

    {
        IntVector vec;
        MinimalisticViewWithSubscription mr_view(vec.begin(), vec.end());
        auto all_view = std::ranges::views::all(mr_view);
        static_assert(contains_host_pointer_v<decltype(all_view)> == false);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(all_view)> == false);
    }
#endif
}

// oneapi::dpl::__ranges::__contains_host_pointer functional with oneapi::dpl::__ranges::zip_view
void
check_contains_host_pointer_in_onedpl_zip_view()
{
    {
        IntVector vec;

        MinimalisticRangeForIntVec mr(vec.begin(), vec.end());
        auto all_view = std::ranges::views::all(mr);

        auto zip_view = oneapi::dpl::__ranges::make_zip_view(all_view, all_view);

        static_assert(contains_host_pointer_v<decltype(all_view)> == true);
        static_assert(contains_host_pointer_v<decltype(zip_view)> == false);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(zip_view)> == true);
    }

    {
        IntVector vec;

        MinimalisticViewWithSubscription mrv(vec.begin(), vec.end());
        auto all_view1 = std::ranges::views::all(mrv);

        MinimalisticRangeForIntVec mr(vec.begin(), vec.end());
        auto all_view2 = std::ranges::views::all(mr);

        auto zip_view = oneapi::dpl::__ranges::make_zip_view(all_view1, all_view2);

        static_assert(contains_host_pointer_v<decltype(all_view1)> == false);
        static_assert(contains_host_pointer_v<decltype(all_view2)> == true);
        static_assert(contains_host_pointer_v<decltype(zip_view)> == false);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(zip_view)> == true);
    }

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

struct SimpleMapForPermutationView
{
    auto operator[](auto arg)
    {
        return arg;
    }
};

struct SimpleFunctorForPermutationView
{
    auto operator()(auto arg)
    {
        return arg;
    }
};

// oneapi::dpl::__ranges::__contains_host_pointer functional with oneapi::dpl::__ranges::permutation_view_simple
void
check_contains_host_pointer_in_onedpl_permutation_view_simple()
{
    {
        IntVector vec;

        MinimalisticRangeForIntVec mr(vec.begin(), vec.end());
        auto all_view = std::ranges::views::all(mr);

        static_assert(oneapi::dpl::__ranges::is_map_view<SimpleMapForPermutationView>::value);
        auto permutation_view =
            oneapi::dpl::__ranges::permutation_view_simple<decltype(all_view), SimpleMapForPermutationView>(
                all_view, SimpleMapForPermutationView{});

        static_assert(contains_host_pointer_v<decltype(all_view)> == true);
        static_assert(contains_host_pointer_v<decltype(permutation_view)> == false);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(permutation_view)> == true);
    }

    {
        IntVector vec;

        MinimalisticRangeForIntVec mr(vec.begin(), vec.end());
        auto all_view = std::ranges::views::all(mr);

        static_assert(oneapi::dpl::__internal::__is_functor<SimpleFunctorForPermutationView>);
        auto permutation_view =
            oneapi::dpl::__ranges::permutation_view_simple<decltype(all_view), SimpleFunctorForPermutationView>(
                all_view, SimpleFunctorForPermutationView{}, 0);

        static_assert(contains_host_pointer_v<decltype(all_view)> == true);
        static_assert(contains_host_pointer_v<decltype(permutation_view)> == false);
        static_assert(contains_host_pointer_on_any_layers_v<decltype(permutation_view)> == true);
    }
}

// oneapi::dpl::__ranges::__contains_host_pointer functional with std::ranges::take_view
void
check_contains_host_pointer_in_std_take_view()
{
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
check_contains_host_pointer_in_std_drop_view()
{
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

    check_contains_host_pointer();
    check_contains_host_pointer_in_onedpl_zip_view();
    check_contains_host_pointer_in_onedpl_permutation_view_simple();
    check_contains_host_pointer_in_std_take_view();
    check_contains_host_pointer_in_std_drop_view();

    bProcessed = true;

#endif // _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(bProcessed);
}
