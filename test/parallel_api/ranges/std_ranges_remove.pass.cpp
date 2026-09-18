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

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING
namespace dpl_ranges = oneapi::dpl::ranges;

template<>
constexpr std::pair<int, int>
test_std_ranges::range_to_verify<std::remove_cvref_t<decltype(dpl_ranges::remove)>>(int total_size, int result_size)
{ 
    return {0, total_size - result_size}; // in the result are the elements to remove
}
#endif

std::int32_t
main()
{
    try
    {
#if _ENABLE_STD_RANGES_TESTING
        using namespace test_std_ranges;

        auto remove_checker = TEST_PREPARE_CALLABLE(std::ranges::remove);

        test_range_algo<0>{get_scan_big_sz()}(dpl_ranges::remove, remove_checker, 0);
        test_range_algo<1>{}(dpl_ranges::remove, remove_checker, 0, proj);
        test_range_algo<2, P2>{}(dpl_ranges::remove, remove_checker, 0, &P2::x);
        test_range_algo<3, P2>{}(dpl_ranges::remove, remove_checker, 0, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING
    }
    catch (const std::exception& exc)
    {
        std::cerr << "Exception occurred in main() of " << __FILE__;
        if (exc.what())
            std::cerr << ": " << exc.what();
        std::cerr << std::endl;

        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "Unknown exception occurred in main() of " << __FILE__ << std::endl;

        return EXIT_FAILURE;
    }

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
