// -*- C++ -*-
//===-- unique_ramp_fixkeep.pass.cpp -------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DIAGNOSTIC BUILD ONLY, DO NOT MERGE. See unique_ramp_common.h.
// The crashing coordinate with an identity write so that no iteration is read-only. Passing here
// both names the mechanism and gives a candidate fix.

#include <cstddef>

#define _ONEDPL_COMPACTION_DIAG_KEEPALIVE 1

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
    run_crash_coordinate<buffer_tag>(exec, plain_unique{}, 4);
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
