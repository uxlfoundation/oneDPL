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

// Covers the partitioned merge path of the hetero set operations at its real threshold, with no override of
// either the threshold or the tile size. Inputs must exceed 2M total elements to reach it, which is above
// TestUtils::get_scan_test_set_max_n(); that cap is deliberately not applied here, because no size within
// it can reach this path. The sizes are chosen so the diagonal count is not a multiple of the tile size,
// leaving a partial final tile.

#include "set_partitioned_common.h"

int
main()
{
    const SizePair sizes[] = {
        {1100000, 1100000}, // 2.2M total: just past the threshold, so the tile count is small
        {2199984, 16}       // the same total, extremely asymmetric
    };

    run_test_set_partitioned(sizes, sizeof(sizes) / sizeof(sizes[0]));

    return TestUtils::done();
}
