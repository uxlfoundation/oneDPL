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

// The hetero set operations switch to a partitioned merge path once the total input size reaches
// _ONEDPL_SET_OP_PARTITION_THRESHOLD, whose default is above every size the other set_* tests use.
// Lowering it is what makes that path reachable from a test; the tile size is left device-derived here,
// so this covers the derived value. set_partitioned_tiles.pass.cpp covers dense tile boundaries.
#define _ONEDPL_SET_OP_PARTITION_THRESHOLD 1024

#include "set_partitioned_common.h"

int
main()
{
    run_test_set_partitioned_small();

    return TestUtils::done();
}
