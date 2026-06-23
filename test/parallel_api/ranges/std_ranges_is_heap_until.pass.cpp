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
struct CustomLess
{
    template <typename T>
    bool
    operator()(const T& lhs, const T& rhs) const
    {
        return lhs < rhs;
    }
};
#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto is_heap_until_checker = TEST_PREPARE_CALLABLE(std::ranges::is_heap_until);

    // Spike at index 42 breaks the max-heap property; is_heap_until stops at that position.
    // big_sz tests the case where the violation is found at a fixed position within a large range.
    auto spike_gen = [](auto i) { return i == 42 ? 1 : -static_cast<int>(i); };
    // Nearly a valid max-heap, but element at index 2000 is corrupted (value 0 > its parent -999).
    // The violation is only detectable near the end of small_size (2025); is_heap_until stops
    // near the end of the range.
    auto late_violation_gen = [](auto i) -> int { return i == 2000 ? 0 : -static_cast<int>(i); };

    // big_sz: stop at a mid-range position (index 42) within a large range (multi-WG device path)
    test_range_algo<0, int, data_in, decltype(spike_gen)>{big_sz}(
        dpl_ranges::is_heap_until, is_heap_until_checker, std::ranges::less{});

    // late violation (index 2000): is_heap_until stops near the end of the range
    test_range_algo<1, int, data_in, decltype(late_violation_gen)>{}(
        dpl_ranges::is_heap_until, is_heap_until_checker, std::ranges::less{}, proj);

    // same late violation with custom comp and P2::x projection
    test_range_algo<2, P2, data_in, decltype(late_violation_gen)>{}(
        dpl_ranges::is_heap_until, is_heap_until_checker, CustomLess{}, &P2::x);

    // ascending default data is a valid min-heap w.r.t. greater; is_heap_until returns end()
    test_range_algo<3, P2>{}(dpl_ranges::is_heap_until, is_heap_until_checker, std::ranges::greater{}, &P2::proj);

    // default overload (comp = less, proj = identity): ascending data, stops at begin()+1
    test_range_algo<4>{}(dpl_ranges::is_heap_until, is_heap_until_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
