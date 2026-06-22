// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto is_heap_checker = TEST_PREPARE_CALLABLE(std::ranges::is_heap);

    // A descending sequence is a max-heap with respect to std::ranges::less, so is_heap is expected
    // to return true; this exercises a full scan of the large device-sized range.
    auto generator = [](auto i) { return -i; };
    test_range_algo<0, int, data_in, decltype(generator)>{big_sz}(
        dpl_ranges::is_heap, is_heap_checker, std::ranges::less{});

    test_range_algo<1>{}(dpl_ranges::is_heap, is_heap_checker, std::ranges::less{}, proj);
    test_range_algo<2, P2>{}(dpl_ranges::is_heap, is_heap_checker, std::ranges::less{}, &P2::x);
    test_range_algo<3, P2>{}(dpl_ranges::is_heap, is_heap_checker, std::ranges::less{}, &P2::proj);

    test_range_algo<4>{}(dpl_ranges::is_heap, is_heap_checker, std::ranges::greater{}, proj);
    test_range_algo<5, P2>{}(dpl_ranges::is_heap, is_heap_checker, std::ranges::greater{}, &P2::x);
    test_range_algo<6, P2>{}(dpl_ranges::is_heap, is_heap_checker, std::ranges::greater{}, &P2::proj);

    test_range_algo<7>{}(dpl_ranges::is_heap, is_heap_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
