// -*- C++ -*-
//===-- unique_ramp_tail.pass.cpp ---------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DIAGNOSTIC BUILD ONLY, DO NOT MERGE. See unique_ramp_common.h.
// The shipping geometry: full segments followed by a short tail, which is the only way the tuned 64 MiB bound
// produces a segment of one, two or three elements. Sweeps the number of full segments preceding the tail.
// If every case here passes, no production input reaches the crash and the deliverable is unaffected.

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
    const std::size_t tails[] = {1, 2, 3};
    const std::size_t segment_sizes[] = {1024, 512, 256, 128, 64, 32, 16, 8, 4, 2};
    for (std::size_t tail : tails)
        for (std::size_t segment_size : segment_sizes)
            if (segment_size > tail)
                run_case<buffer_tag>(exec, plain_unique{}, 1024 + tail, 8, segment_size);
    std::cout << "=== complete ===" << std::endl;
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
