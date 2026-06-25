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

    // Descending values form a max-heap w.r.t. less: element i has value -i,
    // so parent >= child for all nodes.
    auto desc_gen = [](auto i) { return -static_cast<int>(i); };

    // Spike at index 42 breaks the max-heap property; is_heap_until stops at that position.
    // big_sz tests the case where the violation is found at a fixed position within a large range.
    auto spike_gen = [](auto i) { return i == 42 ? 1 : -static_cast<int>(i); };

    constexpr int late_violation_test_sz = 63 * 1024 + 347;
    // Most elements get -i (descending, forming a valid max-heap prefix), but the final 41
    // elements switch to (late_violation_test_sz - i), inserting small positive values
    // whose parents hold large-negative values, breaking the max-heap property near the leaves.
    auto late_violation_gen = [](auto i) {
        int val = static_cast<int>(i);
        return late_violation_test_sz - val > 41 ? -val : late_violation_test_sz - val;
    };

    // Valid min-heap w.r.t. greater but not sorted: element i has value i, except every 17th element
    // (i % 17 == 1) drops to its parent's value (i - 1) / 2, adding parent == child ties along heap paths.
    auto non_desc_heap_gen = [](auto i) { return (i % 17 == 1) ? (i - 1) / 2 : i; };

    // big_sz: stop at a mid-range position (index 42) within a large range (multi-WG device path)
    test_range_algo<0, int, data_in, decltype(spike_gen)>{big_sz}(
        dpl_ranges::is_heap_until, is_heap_until_checker, std::ranges::less{});

    // big_sz: full scan over a large valid max-heap (no violation) exercises the multi-WG device
    // path returning end()
    test_range_algo<1, int, data_in, decltype(desc_gen)>{big_sz}(
        dpl_ranges::is_heap_until, is_heap_until_checker, std::ranges::less{});

    // large range with heap violation near the end: is_heap_until stops before end()
    test_range_algo<2, int, data_in, decltype(late_violation_gen)>{late_violation_test_sz}(
        dpl_ranges::is_heap_until, is_heap_until_checker, std::ranges::less{}, proj);

    // same late-violation data with custom comp and P2::x projection; is_heap_until stops before end()
    test_range_algo<3, P2, data_in, decltype(late_violation_gen)>{late_violation_test_sz}(
        dpl_ranges::is_heap_until, is_heap_until_checker, CustomLess{}, &P2::x);

    // ascending default data is a valid min-heap w.r.t. greater; is_heap_until returns end()
    test_range_algo<4, P2>{}(dpl_ranges::is_heap_until, is_heap_until_checker, std::ranges::greater{}, &P2::proj);

    // valid min-heap w.r.t. greater, not just sorted data: every 17th element (i % 17 == 1) drops to its
    // parent's value (i - 1) / 2, creating parent == child ties that is_heap_until accepts (returns end())
    test_range_algo<5, int, data_in, decltype(non_desc_heap_gen)>{}(
        dpl_ranges::is_heap_until, is_heap_until_checker, std::ranges::greater{});

    // default overload (comp = less, proj = identity): ascending data, stops at begin()+1
    test_range_algo<6>{}(dpl_ranges::is_heap_until, is_heap_until_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
