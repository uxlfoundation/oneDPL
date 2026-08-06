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
template <std::ranges::range R>
auto
get_middle(R&& r)
{
    return std::ranges::begin(r) + std::ranges::size(r) / 2;
}

struct partial_sort_fn
{
    template <typename Policy, typename R, typename... Args>
    std::ranges::borrowed_iterator_t<R>
    operator()(Policy&& exec, R&& r, Args&&... args) const
    {
        auto middle = get_middle(r);
        return oneapi::dpl::ranges::partial_sort(std::forward<Policy>(exec), std::forward<R>(r), middle,
                                                 std::forward<Args>(args)...);
    }
};

template <>
constexpr std::pair<int, int>
test_std_ranges::range_to_verify<partial_sort_fn>(int total_size, int /*result_size*/)
{
    return {0, total_size / 2};
}
#endif //_ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;

    auto partial_sort_algo = partial_sort_fn{};

    auto partial_sort_checker = [](auto&& r, auto&&... args) {
        auto middle = get_middle(r);
        return std::ranges::partial_sort(std::forward<decltype(r)>(r), middle, std::forward<decltype(args)>(args)...);
    };

    std::cerr << "test_range_algo<0>{big_sz}(partial_sort_algo, partial_sort_checker);" << std::endl;
    test_range_algo<0>{big_sz}(partial_sort_algo, partial_sort_checker);
    std::cerr << "test_range_algo<1>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{});" << std::endl;
    test_range_algo<1>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{});

    std::cerr << "test_range_algo<2>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{}, proj);" << std::endl;
    test_range_algo<2>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{}, proj);
    std::cerr << "test_range_algo<3>{}(partial_sort_algo, partial_sort_checker, std::ranges::greater{}, proj);" << std::endl;
    test_range_algo<3>{}(partial_sort_algo, partial_sort_checker, std::ranges::greater{}, proj);

    std::cerr << "test_range_algo<4, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{}, &P2::x);" << std::endl;
    test_range_algo<4, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{}, &P2::x);
    std::cerr << "test_range_algo<5, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::greater{}, &P2::x);" << std::endl;
    test_range_algo<5, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::greater{}, &P2::x);

    std::cerr << "test_range_algo<6, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{}, &P2::proj);" << std::endl;
    test_range_algo<6, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::less{}, &P2::proj);
    std::cerr << "test_range_algo<7, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::greater{}, &P2::proj);" << std::endl;
    test_range_algo<7, P2>{}(partial_sort_algo, partial_sort_checker, std::ranges::greater{}, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
