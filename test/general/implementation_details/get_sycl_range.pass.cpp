// -*- C++ -*-
//===-- get_sycl_range.pass.cpp -------------------------------------------===//
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

#if TEST_DPCPP_BACKEND_PRESENT

#include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>

// Test the compile-time traits __is_copy_direct_v and __is_copy_back_v
// used by __get_sycl_range to determine buffer copy behavior.

void
test_is_copy_direct_v()
{
    // __is_copy_direct_v determines whether data should be copied FROM host TO device
    // when creating a SYCL buffer. It should be true when:
    // - _NoInit is false AND access mode implies reading (read, write, or read_write)
    //
    // Key change in this PR: write mode copies in by default (matching SYCL standard)
    // unless _NoInit=true is specified.

    using read_mode = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::read>;
    using write_mode = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::write>;
    using read_write_mode = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::read_write>;

    using read_mode_no_init = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::read, /*_NoInit=*/true>;
    using write_mode_no_init = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::write, /*_NoInit=*/true>;
    using read_write_mode_no_init =
        oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::read_write, /*_NoInit=*/true>;

    // Test: read mode without no_init -> copy in (true)
    static_assert(read_mode::__is_copy_direct_v<sycl::access::mode::read, false> == true,
                  "read mode without no_init should copy in");

    // Test: read mode with no_init -> no copy in (false)
    static_assert(read_mode_no_init::__is_copy_direct_v<sycl::access::mode::read, true> == false,
                  "read mode with no_init should not copy in");

    // Test: write mode without no_init -> copy in (true)
    static_assert(write_mode::__is_copy_direct_v<sycl::access::mode::write, false> == true,
                  "write mode without no_init should copy in");

    // Test: write mode with no_init -> no copy in (false)
    static_assert(write_mode_no_init::__is_copy_direct_v<sycl::access::mode::write, true> == false,
                  "write mode with no_init should not copy in");

    // Test: read_write mode without no_init -> copy in (true)
    static_assert(read_write_mode::__is_copy_direct_v<sycl::access::mode::read_write, false> == true,
                  "read_write mode without no_init should copy in");

    // Test: read_write mode with no_init -> no copy in (false)
    static_assert(read_write_mode_no_init::__is_copy_direct_v<sycl::access::mode::read_write, true> == false,
                  "read_write mode with no_init should not copy in");
}

void
test_is_copy_back_v()
{
    // __is_copy_back_v determines whether data should be copied FROM device TO host
    // when the SYCL buffer is destroyed. It should be true when:
    // - access mode is write or read_write

    using read_mode = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::read>;
    using write_mode = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::write>;
    using read_write_mode = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::read_write>;

    // Test: read mode -> no copy back (false)
    static_assert(read_mode::__is_copy_back_v<sycl::access::mode::read> == false,
                  "read mode should not copy back");

    // Test: write mode -> copy back (true)
    static_assert(write_mode::__is_copy_back_v<sycl::access::mode::write> == true,
                  "write mode should copy back");

    // Test: read_write mode -> copy back (true)
    static_assert(read_write_mode::__is_copy_back_v<sycl::access::mode::read_write> == true,
                  "read_write mode should copy back");

    // Verify __is_copy_back_v does NOT depend on _NoInit
    using read_mode_no_init = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::read, /*_NoInit=*/true>;
    using write_mode_no_init = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::write, /*_NoInit=*/true>;
    using read_write_mode_no_init =
        oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::read_write, /*_NoInit=*/true>;

    // Test: read mode with no_init -> still no copy back (false)
    static_assert(read_mode_no_init::__is_copy_back_v<sycl::access::mode::read> == false,
                  "read mode with no_init should not copy back");

    // Test: write mode with no_init -> still copy back (true)
    static_assert(write_mode_no_init::__is_copy_back_v<sycl::access::mode::write> == true,
                  "write mode with no_init should still copy back");

    // Test: read_write mode with no_init -> still copy back (true)
    static_assert(read_write_mode_no_init::__is_copy_back_v<sycl::access::mode::read_write> == true,
                  "read_write mode with no_init should still copy back");
}

void
test_traits_use_local_parameters()
{
    // The traits __is_copy_direct_v and __is_copy_back_v are static and depend only on their
    // template parameters (_LocalAccMode, _LocalNoInit), NOT on the struct's template parameters
    // (AccMode, _NoInit). This is important because when processing nested iterators like
    // permutation_iterator, the map iterator is always processed with read mode regardless of
    // the outer access mode.

    // Use a write mode struct but query with read mode parameters (like permutation map iterator)
    using write_no_init_struct = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::write, /*_NoInit=*/true>;

    // Even though the struct is write+no_init, querying with read without no_init should give read behavior
    static_assert(write_no_init_struct::__is_copy_direct_v<sycl::access::mode::read, false> == true,
                  "local read mode should copy in regardless of struct's no_init mode");
    static_assert(write_no_init_struct::__is_copy_back_v<sycl::access::mode::read> == false,
                  "local read mode should not copy back regardless of struct's no_init mode");

    // And querying with write+no_init should give write+no_init behavior
    static_assert(write_no_init_struct::__is_copy_direct_v<sycl::access::mode::write, true> == false,
                  "local write+no_init should not copy in");
    static_assert(write_no_init_struct::__is_copy_back_v<sycl::access::mode::write> == true,
                  "local write mode should copy back");

    // Use a read mode struct but query with write mode parameters
    using read_struct = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::read>;

    static_assert(read_struct::__is_copy_direct_v<sycl::access::mode::write, false> == true,
                  "local write mode should copy in regardless of struct's read mode");
    static_assert(read_struct::__is_copy_back_v<sycl::access::mode::write> == true,
                  "local write mode should copy back regardless of struct's read mode");
}

void
test_default_template_parameter()
{
    // Verify that _NoInit defaults to false
    using write_mode_default = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::write>;
    using write_mode_explicit_false = oneapi::dpl::__ranges::__get_sycl_range<sycl::access::mode::write, false>;

    // Both should have the same copy-in behavior (copy in enabled)
    static_assert(write_mode_default::__is_copy_direct_v<sycl::access::mode::write, false> ==
                      write_mode_explicit_false::__is_copy_direct_v<sycl::access::mode::write, false>,
                  "default _NoInit should be false");
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_is_copy_direct_v();
    test_is_copy_back_v();
    test_traits_use_local_parameters();
    test_default_template_parameter();
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
