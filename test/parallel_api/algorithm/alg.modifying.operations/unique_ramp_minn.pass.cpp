// -*- C++ -*-
//===-- unique_ramp_minn.pass.cpp ---------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DIAGNOSTIC BUILD ONLY, DO NOT MERGE. See unique_ramp_common.h.
// The smallest input that crashes at one element per segment and an 8:1 drop rate. A small one would be a
// reproducer worth trying to bring up outside CI.

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
    const std::size_t sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024, 4096};
    for (std::size_t n : sizes)
        run_case<buffer_tag>(exec, plain_unique{}, n, 8, 1);
    std::cout << "=== complete ===" << std::endl;
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
