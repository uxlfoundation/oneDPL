// -*- C++ -*-
//===-- unique_ramp_droprate.pass.cpp ---------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DIAGNOSTIC BUILD ONLY, DO NOT MERGE. See unique_ramp_common.h.
// At one element per segment, the smallest run length that crashes. run_length - 1 segments in a row contribute
// no survivor, so the first crashing run length is the length of zero-survivor run that the loop cannot take.

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
    const std::size_t run_lengths[] = {2, 3, 4, 5, 6, 8, 16, 64};
    for (std::size_t run_length : run_lengths)
        run_case<buffer_tag>(exec, plain_unique{}, 1024, run_length, 1);
    std::cout << "=== complete ===" << std::endl;
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
