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

    auto is_heap_checker = TEST_PREPARE_CALLABLE(std::ranges::is_heap);

    // Descending values form a max-heap w.r.t. less: element i has value -i,
    // so parent >= child for all nodes.
    auto desc_gen = [](auto i) { return -static_cast<int>(i); };

    constexpr int late_violation_test_sz = 63 * 1024 + 347;
    // Strictly descending data (a valid max-heap w.r.t. less): element i gets
    // late_violation_test_sz - i until that would fall to 41 or below, then the
    // final 41 elements switch to -i, pushing the smallest values to the tail.
    auto late_violation_gen = [](auto i) {
        int val = static_cast<int>(i);
        return late_violation_test_sz - val > 41 ? late_violation_test_sz - val : -val;
    };

    // Valid min-heap w.r.t. greater but not sorted: element i has value i, except every 17th element
    // (i % 17 == 1) drops to its parent's value (i - 1) / 2, adding parent == child ties along heap paths.
    auto non_desc_heap_gen = [](auto i) { return (i % 17 == 1) ? (i - 1) / 2 : i; };

    // --- returns true ---

    // big_sz: exercises the multi-work-group device path on the true (full-scan) case
    test_range_algo<0, int, data_in, decltype(desc_gen)>{big_sz}(
        dpl_ranges::is_heap, is_heap_checker, std::ranges::less{});

    // custom comp + P2::x (member-data projection); descending x values form a max-heap
    test_range_algo<1, P2, data_in, decltype(desc_gen)>{}(
        dpl_ranges::is_heap, is_heap_checker, CustomLess{}, &P2::x);

    // ascending default data is a valid min-heap w.r.t. greater; member-function projection
    test_range_algo<2, P2>{}(dpl_ranges::is_heap, is_heap_checker, std::ranges::greater{}, &P2::proj);

    // valid min-heap w.r.t. greater, not just sorted data: every 17th element (i % 17 == 1) drops to its
    // parent's value (i - 1) / 2, creating parent == child ties that is_heap must accept (returns true)
    test_range_algo<21, int, data_in, decltype(non_desc_heap_gen)>{}(dpl_ranges::is_heap, is_heap_checker,
                                                                     std::ranges::greater{});

    // --- returns false ---

    // large valid max-heap: data stays descending to the very end, so is_heap scans the whole range and returns true
    test_range_algo<3, int, data_in, decltype(late_violation_gen)>{late_violation_test_sz}(
        dpl_ranges::is_heap, is_heap_checker, std::ranges::less{}, proj);

    // same full-range valid max-heap with custom comp and P2::x projection
    test_range_algo<4, P2, data_in, decltype(late_violation_gen)>{late_violation_test_sz}(
        dpl_ranges::is_heap, is_heap_checker, CustomLess{}, &P2::x);

    // default overload (comp = less, proj = identity): ascending data violates max-heap property
    test_range_algo<5>{}(dpl_ranges::is_heap, is_heap_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
