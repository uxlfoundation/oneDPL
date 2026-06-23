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
    // Nearly a valid max-heap, but element at index 2000 is corrupted (value 0 > its parent -999).
    // The violation is only detectable near the end of small_size (2025), forcing nearly the full
    // range to be scanned before reporting false.
    auto late_violation_gen = [](auto i) -> int { return i == 2000 ? 0 : -static_cast<int>(i); };

    // --- returns true ---

    // big_sz: exercises the multi-work-group device path on the true (full-scan) case
    test_range_algo<0, int, data_in, decltype(desc_gen)>{big_sz}(
        dpl_ranges::is_heap, is_heap_checker, std::ranges::less{});

    // custom comp + P2::x (member-data projection); descending x values form a max-heap
    test_range_algo<1, P2, data_in, decltype(desc_gen)>{}(
        dpl_ranges::is_heap, is_heap_checker, CustomLess{}, &P2::x);

    // ascending default data is a valid min-heap w.r.t. greater; member-function projection
    test_range_algo<2, P2>{}(dpl_ranges::is_heap, is_heap_checker, std::ranges::greater{}, &P2::proj);

    // --- returns false ---

    // heap violation at index 2000 (near the end); forces nearly the full range to be checked
    test_range_algo<3, int, data_in, decltype(late_violation_gen)>{}(
        dpl_ranges::is_heap, is_heap_checker, std::ranges::less{}, proj);

    // same late violation with custom comp and P2::x projection
    test_range_algo<4, P2, data_in, decltype(late_violation_gen)>{}(
        dpl_ranges::is_heap, is_heap_checker, CustomLess{}, &P2::x);

    // default overload (comp = less, proj = identity): ascending data violates max-heap property
    test_range_algo<5>{}(dpl_ranges::is_heap, is_heap_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
