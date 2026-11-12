// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test for a DPC++ runtime bug and its oneDPL workaround, _ONEDPL_SYCL_RANGED_ACCESSOR_NO_INIT_BROKEN.
// oneDPL requests no_init for the output accessor of an algorithm like fill. Per the SYCL specification, on a ranged
// accessor no_init must leave the elements outside the accessor's range untouched, but affected runtimes discard the
// whole buffer. For those versions oneDPL drops no_init from ranged accessors; this test fails if the workaround is
// disabled for a runtime that still has the bug. The bug is only observable on devices without host unified memory,
// e.g. discrete GPUs, or with SYCL_HOST_UNIFIED_MEMORY=0.

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include <vector>
#endif

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    constexpr std::size_t n = 1000, lo = 100, hi = 700;

    // No host pointer, and filled through a host_accessor, so the data must be copied to the device on first use.
    sycl::buffer<int> buf{sycl::range<1>(n)};
    {
        sycl::host_accessor acc{buf, sycl::write_only};
        for (std::size_t i = 0; i < n; ++i)
            acc[i] = 42;
    }

    oneapi::dpl::fill(TestUtils::get_dpcpp_test_policy(), oneapi::dpl::begin(buf) + lo, oneapi::dpl::begin(buf) + hi,
                      7);

    std::vector<int> expected(n, 42);
    std::fill(expected.begin() + lo, expected.begin() + hi, 7);

    sycl::host_accessor acc{buf, sycl::read_only};
    EXPECT_EQ_N(expected.begin(), acc.begin(), n, "elements outside of the written sub-range were not preserved");
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
