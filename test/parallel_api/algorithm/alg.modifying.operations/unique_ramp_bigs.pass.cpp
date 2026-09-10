// -*- C++ -*-
//===-- unique_ramp_bigs.pass.cpp ---------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DIAGNOSTIC BUILD ONLY, DO NOT MERGE. See unique_ramp_common.h.
// The same zero-survivor runs at more than one element per segment. If nothing here crashes, a segment_size of
// exactly 1 is necessary; if s=2 crashes at a long enough run, the zero-survivor run alone is the mechanism.

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
    const std::size_t segment_sizes[] = {8, 4, 2};
    const std::size_t run_lengths[] = {2, 4, 8, 16, 32, 64, 256};
    for (std::size_t segment_size : segment_sizes)
        for (std::size_t run_length : run_lengths)
            run_case<buffer_tag>(exec, plain_unique{}, 1024, run_length, segment_size);
    std::cout << "=== complete ===" << std::endl;
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
