// -*- C++ -*-
//===-- unique_ramp_trace.pass.cpp ---------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DIAGNOSTIC BUILD ONLY, DO NOT MERGE. See unique_ramp_common.h.
// The crashing case with the library loop tracing every iteration, so the last line names the iteration and
// which of the two submissions the process died in. One case only, to stay well under ctest's output cap.

#include <cstddef>

#define _ONEDPL_COMPACTION_TRACE 1

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
    run_case<buffer_tag>(exec, plain_unique{}, 1024, 8, 1);
    std::cout << "=== complete ===" << std::endl;
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
