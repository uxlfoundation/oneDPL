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

// NOT FOR MERGE. Diagnostic: identical to set_partitioned_path.pass except that every diagonal searches the
// whole input instead of the box the partitioning pass established for its tile. No work item then reads the
// balanced path temporary while siblings write it, which removes the read side of that overlap, and it also
// takes the tile bounds themselves out of the picture.
#define _ONEDPL_SET_OP_PARTITION_THRESHOLD 1024
#define _ONEDPL_SET_OP_DIAG_FULL_RANGE_BOUNDS 1

#include "set_partitioned_common.h"

int
main()
{
    run_test_set_partitioned_small();

    return TestUtils::done();
}
