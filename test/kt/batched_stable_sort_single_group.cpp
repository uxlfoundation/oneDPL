// -*- C++ -*-
//===-- batched_stable_sort_single_group.cpp ------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// radix_single_group_tag, with segments which fit in a work-group tile
#define TEST_SINGLE_GROUP 1
#include "batched_stable_sort.cpp"
