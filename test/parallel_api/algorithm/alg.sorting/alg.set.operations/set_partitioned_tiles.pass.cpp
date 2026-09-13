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

// Same coverage as set_partitioned_path.pass.cpp, but with the tile of the partitioned merge path forced
// to three diagonals. The device-derived tile spans more elements than the whole input at every size the
// suite uses, so without this override no test crosses a tile boundary; with it every input crosses
// hundreds, exercising the balanced path star correction and the bounds decode at each one.
#define _ONEDPL_SET_OP_PARTITION_THRESHOLD 1024
#define _ONEDPL_SET_OP_PARTITION_TILE_DIAGONALS 3

#include "set_partitioned_common.h"

int
main()
{
    run_test_set_partitioned_small();

    return TestUtils::done();
}
