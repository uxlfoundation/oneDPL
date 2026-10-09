// -*- C++ -*-
//===-- batched_stable_sort_merge_wg_resident.cpp -------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "batched_stable_sort.h"

template <bool IsAscending>
using SortConfig = kt::gpu::merge_single_group_tag;

int
main()
{
    return test<SortConfig, /*WgResidentOnly=*/true>();
}
