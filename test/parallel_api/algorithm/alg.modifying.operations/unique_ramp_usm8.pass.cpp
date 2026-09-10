// -*- C++ -*-
//===-- unique_ramp_usm8.pass.cpp -----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DIAGNOSTIC BUILD ONLY, DO NOT MERGE. See unique_ramp_common.h.
// The crashing coordinate over a USM device pointer. unique_ramp_usm ran the same memory path at a 2:1 drop rate,
// which run 34440826733 showed cannot crash at any segment size, so the memory path is in fact untested.

#include <cstddef>

namespace segment_size_override
{
std::size_t bytes = 64 * 1024 * 1024;
}
#define _ONEDPL_COMPACTION_SEGMENT_SIZE_BYTES (::segment_size_override::bytes)

#include "unique_ramp_common.h"

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto exec = TestUtils::get_dpcpp_test_policy();
    run_crash_coordinate<usm_device_tag>(exec, plain_unique{}, 4);
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
