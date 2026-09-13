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

// NOT FOR MERGE. Diagnostic: identical to set_partitioned_path.pass except that the reduce kernel no longer
// stores the last diagonal, which is the only entry of the balanced path temporary it both writes and reads.
// This removes the write side of that overlap; set_partitioned_fullbounds.pass removes the read side.
#define _ONEDPL_SET_OP_PARTITION_THRESHOLD 1024
#define _ONEDPL_SET_OP_DIAG_NO_LAST_STORE 1

#include "set_partitioned_common.h"

int
main()
{
    run_test_set_partitioned_small();

    return TestUtils::done();
}
