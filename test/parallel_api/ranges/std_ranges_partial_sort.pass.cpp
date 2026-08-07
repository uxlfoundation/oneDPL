// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING

enum class middle_pos
{
    first,
    half,
    last
};

template <middle_pos Pos, std::ranges::range R>
std::ranges::borrowed_iterator_t<R>
get_middle(R&& r)
{
    if constexpr (Pos == middle_pos::first)
        return std::ranges::begin(r);
    else if constexpr (Pos == middle_pos::half)
        return std::ranges::begin(r) + std::ranges::size(r) / 2;
    else
        return std::ranges::begin(r) + std::ranges::size(r);
}

template <middle_pos Pos>
struct partial_sort_fn
{
    template <typename Policy, typename R, typename... Args>
    std::ranges::borrowed_iterator_t<R>
    operator()(Policy&& exec, R&& r, Args&&... args) const
    {
        auto middle = get_middle<Pos>(r);
        return oneapi::dpl::ranges::partial_sort(std::forward<Policy>(exec), std::forward<R>(r), middle,
                                                 std::forward<Args>(args)...);
    }
};

template <>
constexpr std::pair<int, int>
test_std_ranges::range_to_verify<partial_sort_fn<middle_pos::first>>(int /*total_size*/, int /*result_size*/)
{
    return {0, 0};
}

template <>
constexpr std::pair<int, int>
test_std_ranges::range_to_verify<partial_sort_fn<middle_pos::half>>(int total_size, int /*result_size*/)
{
    return {0, total_size / 2};
}

template <>
constexpr std::pair<int, int>
test_std_ranges::range_to_verify<partial_sort_fn<middle_pos::last>>(int total_size, int /*result_size*/)
{
    return {0, total_size};
}
#endif //_ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;

    auto partial_sort_algo = partial_sort_fn<middle_pos::half>{};

    auto partial_sort_checker = [](auto&& r, auto&&... args) -> std::ranges::borrowed_iterator_t<decltype(r)> {
        auto middle = get_middle<middle_pos::half>(r);
        return std::ranges::partial_sort(std::forward<decltype(r)>(r), middle, std::forward<decltype(args)>(args)...);
    };

    test_range_algo<0>{big_sz}(partial_sort_algo, partial_sort_checker);
    test_range_algo<1>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{});

    test_range_algo<2>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{}, proj);
    test_range_algo<3>{}(partial_sort_algo, partial_sort_checker, std::ranges::greater{}, proj);

    test_range_algo<4, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{}, &P2::x);
    test_range_algo<5, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::greater{}, &P2::x);

    test_range_algo<6, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{}, &P2::proj);
    test_range_algo<7, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::greater{}, &P2::proj);

    // Boundary case: middle == begin(r), the algorithm is a no-op
    auto partial_sort_none_algo = partial_sort_fn<middle_pos::first>{};
    auto partial_sort_none_checker = [](auto&& r, auto&&... args) -> std::ranges::borrowed_iterator_t<decltype(r)> {
        auto middle = get_middle<middle_pos::first>(r);
        return std::ranges::partial_sort(std::forward<decltype(r)>(r), middle, std::forward<decltype(args)>(args)...);
    };

    test_range_algo<8>{big_sz}(partial_sort_none_algo, partial_sort_none_checker);
    test_range_algo<9>{}(partial_sort_none_algo, partial_sort_none_checker, std::ranges::greater{}, proj);

    // Boundary case: middle == end(r), the whole range is sorted
    auto partial_sort_all_algo = partial_sort_fn<middle_pos::last>{};
    auto partial_sort_all_checker = [](auto&& r, auto&&... args) {
        auto middle = get_middle<middle_pos::last>(r);
        return std::ranges::partial_sort(std::forward<decltype(r)>(r), middle, std::forward<decltype(args)>(args)...);
    };

    test_range_algo<10>{big_sz}(partial_sort_all_algo, partial_sort_all_checker);
    test_range_algo<11>{}(partial_sort_all_algo, partial_sort_all_checker, std::ranges::greater{}, proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
