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

// NOT FOR MERGE. Diagnostic: the partitioned path with a one-diagonal tile. Every diagonal is then a tile
// boundary, so the partitioning pass computes and stores all of them with full-range bounds and the reduce
// kernel only decodes -- it never recomputes a balanced path, never writes the temporary, and never asks
// __get_bounds_partitioned for a box. Uses unmodified library code. If this still drops an output element,
// the defect is in the partitioning pass or in the emit machinery, not in the tile bounds or the recompute.
#define _ONEDPL_SET_OP_PARTITION_THRESHOLD 1024
#define _ONEDPL_SET_OP_PARTITION_TILE_DIAGONALS 1

#include "set_partitioned_common.h"

int
main()
{
    run_test_set_partitioned_small();

    return TestUtils::done();
}
